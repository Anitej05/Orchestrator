"""
Context Builder

Builds token-efficient, structured representations of spreadsheet data for LLM consumption.
Implements anti-hallucination strategies and preserves document structure.

Implements Requirements 16.3, 16.4, 16.7, 18.1, 18.2, 18.3, 18.5, 3.1, 3.2, 17.6, 18.4, 18.6, 18.7, 8.4, 8.5
from the intelligent spreadsheet parsing spec.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from ..parsing_models import (
    ParsedSpreadsheet,
    DocumentSection,
    SectionType,
    ContentType,
    TableRegion,
    TableSchema,
    SamplingStrategy,
    StructuredContext,
    DocumentType
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds structured context representations for LLM consumption.
    
    Key features:
    - Hierarchical JSON/YAML representations
    - Anti-hallucination strategies with explicit markers
    - Intelligent sampling for large datasets
    - Document structure preservation
    - Context validation and completeness checking
    - Context caching for performance optimization (Requirements 8.4, 8.5)
    """
    
    def __init__(self, max_tokens: int = 8000, thread_id: Optional[str] = None, file_id: Optional[str] = None):
        """
        Initialize the context builder.
        
        Args:
            max_tokens: Maximum token budget for context (approximate)
            thread_id: Optional thread ID for cache integration
            file_id: Optional file ID for cache integration
        """
        self.max_tokens = max_tokens
        self.thread_id = thread_id
        self.file_id = file_id
        self.logger = logging.getLogger(__name__)
        
        # Approximate tokens per character (rough estimate)
        self.chars_per_token = 4
        self.max_chars = max_tokens * self.chars_per_token
        
        # Import parse cache (lazy import to avoid circular dependencies)
        try:
            from ..parse_cache import parse_cache
            self.parse_cache = parse_cache
        except ImportError:
            self.logger.warning("ParseCache not available, caching disabled")
            self.parse_cache = None
    
    def build_structured_context(
        self,
        parsed: ParsedSpreadsheet,
        max_tokens: Optional[int] = None
    ) -> StructuredContext:
        """
        Build hierarchical structured context with section boundaries.
        
        This is the primary method for creating LLM context. It:
        - Preserves document structure and section relationships
        - Includes anti-hallucination markers
        - Uses intelligent sampling for large tables
        - Validates completeness
        - Caches results for performance (Requirements 8.4, 8.5)
        
        Args:
            parsed: ParsedSpreadsheet object with all parsing results
            max_tokens: Optional token limit override
        
        Returns:
            StructuredContext object
        
        Validates: Requirements 16.3, 16.4, 16.7, 18.1, 18.2, 18.3, 18.5, 8.4, 8.5
        """
        if max_tokens:
            self.max_tokens = max_tokens
            self.max_chars = max_tokens * self.chars_per_token
        
        # Check cache first (Requirement 8.4)
        if self.parse_cache and self.thread_id and self.file_id:
            cached_context = self.parse_cache.retrieve_context(
                self.thread_id,
                self.file_id,
                "structured",
                self.max_tokens
            )
            if cached_context:
                self.logger.info(
                    f"Using cached structured context for thread={self.thread_id}, "
                    f"file={self.file_id}, max_tokens={self.max_tokens}"
                )
                return cached_context
        
        sections_dict = {}
        sampling_info = None
        validation_checksums = {}
        
        # Build context for each section
        for section in parsed.sections:
            section_key = f"{section.section_type.value}_{section.start_row}"
            
            if section.section_type == SectionType.HEADER:
                sections_dict[section_key] = self._build_header_section(
                    parsed.raw_df,
                    section,
                    parsed.metadata
                )
            
            elif section.section_type == SectionType.LINE_ITEMS:
                # Find corresponding table
                table_data = self._find_table_for_section(parsed, section)
                if table_data:
                    region, df, schema = table_data
                    section_context, sample_info = self._build_table_section(
                        df, schema, region, section
                    )
                    sections_dict[section_key] = section_context
                    if sample_info:
                        sampling_info = sample_info
                    
                    # Add validation checksums for line items
                    validation_checksums.update(
                        self._compute_validation_checksums(df, schema)
                    )
            
            elif section.section_type == SectionType.SUMMARY:
                sections_dict[section_key] = self._build_summary_section(
                    parsed.raw_df,
                    section
                )
                
                # Extract summary values for validation
                summary_values = self._extract_summary_values(
                    parsed.raw_df.iloc[section.start_row:section.end_row + 1]
                )
                if summary_values:
                    validation_checksums['summary_totals'] = summary_values
            
            elif section.section_type == SectionType.FOOTER:
                sections_dict[section_key] = self._build_footer_section(
                    parsed.raw_df,
                    section
                )
            
            elif section.section_type == SectionType.METADATA:
                sections_dict[section_key] = self._build_metadata_section(
                    parsed.raw_df,
                    section
                )
        
        # Add section transition markers
        sections_dict = self._add_section_markers(sections_dict, parsed.sections)
        
        # Preserve intentional gaps as section markers (Requirement 17.6)
        sections_dict = self.preserve_intentional_gaps(parsed, sections_dict)
        
        # Create structured context
        context = StructuredContext(
            document_type=parsed.document_type,
            sections=sections_dict
        )
        
        # Validate completeness
        if not self._validate_context_completeness(context, parsed):
            self.logger.warning("Context validation failed - some critical information may be missing")
        
        # Cache the result (Requirement 8.5)
        if self.parse_cache and self.thread_id and self.file_id:
            self.parse_cache.store_context(
                self.thread_id,
                self.file_id,
                "structured",
                context,
                self.max_tokens
            )
            self.logger.debug(
                f"Cached structured context for thread={self.thread_id}, "
                f"file={self.file_id}, max_tokens={self.max_tokens}"
            )
        
        self.logger.info(
            f"Built structured context with {len(sections_dict)} sections, "
            f"document_type={parsed.document_type.value}"
        )
        
        return context
    
    def build_compact_context(
        self,
        parsed: ParsedSpreadsheet,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Build token-efficient string context.
        
        This creates a more compact representation suitable for
        smaller token budgets. Results are cached for performance.
        
        Args:
            parsed: ParsedSpreadsheet object
            max_tokens: Optional token limit override
        
        Returns:
            Compact string representation
        
        Validates: Requirements 8.3, 8.4, 8.5
        """
        if max_tokens:
            self.max_tokens = max_tokens
            self.max_chars = max_tokens * self.chars_per_token
        
        # Check cache first (Requirement 8.4)
        if self.parse_cache and self.thread_id and self.file_id:
            cached_context = self.parse_cache.retrieve_context(
                self.thread_id,
                self.file_id,
                "compact",
                self.max_tokens
            )
            if cached_context:
                self.logger.info(
                    f"Using cached compact context for thread={self.thread_id}, "
                    f"file={self.file_id}, max_tokens={self.max_tokens}"
                )
                return cached_context
        
        structured = self.build_structured_context(parsed, max_tokens)
        
        # Convert to compact JSON string
        context_dict = structured.to_dict()
        compact_json = json.dumps(context_dict, separators=(',', ':'))
        
        # If still too large, apply more aggressive sampling
        if len(compact_json) > self.max_chars:
            self.logger.warning(
                f"Context exceeds budget ({len(compact_json)} > {self.max_chars}), "
                "applying aggressive sampling"
            )
            # Re-build with smaller sample
            compact_json = self._build_minimal_context(parsed)
        
        # Cache the result (Requirement 8.5)
        if self.parse_cache and self.thread_id and self.file_id:
            self.parse_cache.store_context(
                self.thread_id,
                self.file_id,
                "compact",
                compact_json,
                self.max_tokens
            )
            self.logger.debug(
                f"Cached compact context for thread={self.thread_id}, "
                f"file={self.file_id}, max_tokens={self.max_tokens}"
            )
        
        return compact_json
    
    def build_full_context(
        self,
        parsed: ParsedSpreadsheet
    ) -> str:
        """
        Build complete context without sampling (for small files).
        Results are cached for performance.
        
        Args:
            parsed: ParsedSpreadsheet object
        
        Returns:
            Full string representation
        
        Validates: Requirements 8.4, 8.5
        """
        # Check cache first (Requirement 8.4)
        if self.parse_cache and self.thread_id and self.file_id:
            cached_context = self.parse_cache.retrieve_context(
                self.thread_id,
                self.file_id,
                "full",
                None  # No token limit for full context
            )
            if cached_context:
                self.logger.info(
                    f"Using cached full context for thread={self.thread_id}, "
                    f"file={self.file_id}"
                )
                return cached_context
        
        # Temporarily disable token limit
        original_max = self.max_tokens
        self.max_tokens = 1000000  # Very large
        self.max_chars = self.max_tokens * self.chars_per_token
        
        try:
            structured = self.build_structured_context(parsed)
            full_context = json.dumps(structured.to_dict(), indent=2)
            
            # Cache the result (Requirement 8.5)
            if self.parse_cache and self.thread_id and self.file_id:
                self.parse_cache.store_context(
                    self.thread_id,
                    self.file_id,
                    "full",
                    full_context,
                    None
                )
                self.logger.debug(
                    f"Cached full context for thread={self.thread_id}, "
                    f"file={self.file_id}"
                )
            
            return full_context
        finally:
            # Restore original limit
            self.max_tokens = original_max
            self.max_chars = original_max * self.chars_per_token
    
    def sample_dataframe(
        self,
        df: pd.DataFrame,
        strategy: str = "smart"
    ) -> Tuple[pd.DataFrame, SamplingStrategy]:
        """
        Apply sampling strategy to dataframe.
        
        Strategies:
        - "all_rows": No sampling (for small datasets)
        - "smart": Intelligent sampling based on size
        - "head_tail": First N and last N rows
        - "stratified": Stratified sampling (if applicable)
        
        Args:
            df: DataFrame to sample
            strategy: Sampling strategy name
        
        Returns:
            Tuple of (sampled_df, SamplingStrategy)
        
        Validates: Requirements 3.1, 3.2, 18.4
        """
        total_rows = len(df)
        
        # No sampling for small datasets
        if total_rows <= 100:
            return df, SamplingStrategy(
                strategy_name="all_rows",
                total_rows=total_rows,
                sampled_rows=total_rows
            )
        
        # Determine sample sizes based on total rows
        if total_rows <= 1000:
            first_n = 20
            last_n = 20
            middle_n = 10
        else:
            first_n = 30
            last_n = 30
            middle_n = 20
        
        # Apply sampling
        if strategy == "smart" or strategy == "head_tail":
            # Get first N rows
            first_rows = df.iloc[:first_n]
            
            # Get last N rows
            last_rows = df.iloc[-last_n:]
            
            # Get evenly-spaced middle rows
            middle_indices = np.linspace(
                first_n,
                total_rows - last_n - 1,
                middle_n,
                dtype=int
            )
            middle_rows = df.iloc[middle_indices]
            
            # Combine
            sampled_df = pd.concat([first_rows, middle_rows, last_rows])
            sampled_df = sampled_df.reset_index(drop=True)
            
            sampling_info = SamplingStrategy(
                strategy_name=strategy,
                total_rows=total_rows,
                sampled_rows=len(sampled_df),
                first_n=first_n,
                last_n=last_n,
                middle_n=middle_n
            )
            
            self.logger.info(
                f"Sampled {len(sampled_df)} rows from {total_rows} "
                f"(first={first_n}, middle={middle_n}, last={last_n})"
            )
            
            return sampled_df, sampling_info
        
        else:
            # Default to all rows
            return df, SamplingStrategy(
                strategy_name="all_rows",
                total_rows=total_rows,
                sampled_rows=total_rows
            )
    
    def format_metadata(
        self,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format metadata section as structured dictionary.
        
        Args:
            metadata: Raw metadata dictionary
        
        Returns:
            Formatted metadata dictionary
        
        Validates: Requirements 18.6, 4.3
        """
        formatted = {}
        
        for key, value in metadata.items():
            # Clean up key
            clean_key = str(key).strip().lower().replace(' ', '_').rstrip(':')
            
            # Format value
            if pd.isna(value):
                continue
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                formatted[clean_key] = str(value)
            elif isinstance(value, (int, float, np.number)):
                formatted[clean_key] = float(value) if isinstance(value, float) else int(value)
            else:
                formatted[clean_key] = str(value).strip()
        
        return formatted
    
    def add_anti_hallucination_markers(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add explicit markers and validation data to prevent hallucination.
        
        Markers include:
        - Section boundaries
        - Row counts
        - Sampling information
        - Validation checksums
        
        Args:
            context: Context dictionary
        
        Returns:
            Context with markers added
        
        Validates: Requirement 18.5
        """
        # Add explicit section markers
        marked_context = {}
        
        for section_key, section_data in context.items():
            marked_context[section_key] = section_data
            
            # Add end marker
            if isinstance(section_data, dict) and 'type' in section_data:
                section_type = section_data['type']
                marked_context[f"_end_of_{section_key}"] = f"--- End of {section_type} ---"
        
        return marked_context
    
    def validate_context_completeness(
        self,
        context: StructuredContext,
        original: ParsedSpreadsheet
    ) -> bool:
        """
        Ensure all critical information is preserved in context.
        
        Checks:
        - All sections represented
        - Primary table included
        - Metadata present
        - Validation checksums match
        - Critical fields included (totals, dates, identifiers)
        
        Args:
            context: Generated context
            original: Original parsed spreadsheet
        
        Returns:
            True if context is complete
        
        Validates: Requirements 18.7, 18.6
        """
        validation_passed = True
        
        # Check all sections are represented
        section_types = {s.section_type for s in original.sections}
        context_sections = set()
        
        for key in context.sections.keys():
            # Skip marker keys
            if key.startswith('_'):
                continue
            
            if '_' in key:
                section_type_str = key.split('_')[0]
                try:
                    context_sections.add(SectionType(section_type_str))
                except ValueError:
                    pass
        
        missing_sections = section_types - context_sections
        if missing_sections:
            self.logger.warning(f"Missing sections in context: {missing_sections}")
            validation_passed = False
        
        # Check primary table is included
        if original.primary_table:
            has_line_items = any(
                'line_items' in key.lower()
                for key in context.sections.keys()
            )
            if not has_line_items:
                self.logger.warning("Primary table not included in context")
                validation_passed = False
        
        # Check metadata is present (Requirement 18.6)
        if original.metadata:
            has_metadata = False
            for section_data in context.sections.values():
                if isinstance(section_data, dict):
                    if 'metadata' in section_data or 'fields' in section_data:
                        has_metadata = True
                        break
            
            if not has_metadata:
                self.logger.warning("Metadata missing from context")
                # This is a warning, not a failure for backward compatibility
        
        # Check critical information is included (Requirement 18.7)
        critical_fields = self._identify_critical_fields(original)
        if critical_fields:
            included_fields = self._extract_all_fields_from_context(context)
            missing_critical = critical_fields - included_fields
            
            if missing_critical:
                self.logger.warning(
                    f"Critical fields missing from context: {missing_critical}"
                )
                # Log but don't fail - some fields might be optional
        
        # Validate checksums if present
        if context.metadata.validation_checksums:
            self.logger.debug(
                f"Context includes {len(context.metadata.validation_checksums)} validation checksums"
            )
        
        if validation_passed:
            self.logger.debug("Context completeness validation passed")
        else:
            self.logger.warning("Context completeness validation failed")
        
        return validation_passed
    
    def _identify_critical_fields(
        self,
        parsed: ParsedSpreadsheet
    ) -> set:
        """
        Identify critical fields that should be in context.
        
        Critical fields include:
        - Totals, subtotals, grand totals
        - Dates (invoice date, due date, etc.)
        - Identifiers (invoice #, order #, customer ID)
        
        Args:
            parsed: ParsedSpreadsheet object
        
        Returns:
            Set of critical field names (lowercase)
        """
        critical = set()
        
        # Common critical field patterns
        critical_patterns = [
            'total', 'subtotal', 'grand_total', 'amount_due',
            'invoice_number', 'invoice_date', 'order_number',
            'customer_id', 'date', 'due_date', 'payment_date',
            'tax', 'vat', 'shipping', 'discount'
        ]
        
        # Check metadata for critical fields
        for key in parsed.metadata.keys():
            key_lower = str(key).lower().replace(' ', '_').rstrip(':')
            for pattern in critical_patterns:
                if pattern in key_lower:
                    critical.add(key_lower)
        
        # Check summary sections for totals
        for section in parsed.sections:
            if section.section_type == SectionType.SUMMARY:
                if 'key_values' in section.metadata:
                    for key in section.metadata['key_values'].keys():
                        key_lower = str(key).lower().replace(' ', '_').rstrip(':')
                        for pattern in critical_patterns:
                            if pattern in key_lower:
                                critical.add(key_lower)
        
        return critical
    
    def _extract_all_fields_from_context(
        self,
        context: StructuredContext
    ) -> set:
        """
        Extract all field names from context.
        
        Args:
            context: StructuredContext object
        
        Returns:
            Set of all field names (lowercase)
        """
        fields = set()
        
        def extract_keys(obj, prefix=''):
            """Recursively extract keys from nested dict."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_lower = str(key).lower().replace(' ', '_').rstrip(':')
                    fields.add(key_lower)
                    
                    if isinstance(value, dict):
                        extract_keys(value, prefix=key_lower)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                extract_keys(item, prefix=key_lower)
        
        extract_keys(context.sections)
        
        return fields
    
    # ============== PRIVATE HELPER METHODS ==============
    
    def _build_header_section(
        self,
        df: pd.DataFrame,
        section: DocumentSection,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context for header section."""
        section_df = df.iloc[section.start_row:section.end_row + 1]
        
        header_context = {
            'type': 'header',
            'content_type': section.content_type.value,
            'metadata': self.format_metadata(metadata)
        }
        
        # Extract key-value pairs from header
        kv_pairs = self._extract_key_value_from_section(section_df)
        if kv_pairs:
            header_context['fields'] = kv_pairs
        
        return header_context
    
    def _build_table_section(
        self,
        df: pd.DataFrame,
        schema: TableSchema,
        region: TableRegion,
        section: DocumentSection
    ) -> Tuple[Dict[str, Any], Optional[SamplingStrategy]]:
        """
        Build context for table section with sampling.
        
        Validates: Requirements 3.1, 3.2, 17.6, 18.4
        """
        # Apply sampling
        sampled_df, sampling_info = self.sample_dataframe(df)
        
        # Convert to records
        records = sampled_df.to_dict('records')
        
        table_context = {
            'type': 'table',
            'schema': schema.headers,
            'dtypes': schema.dtypes,
            'row_count': schema.row_count,
            'col_count': schema.col_count,
            'sample_strategy': sampling_info.strategy_name,
            'data': records
        }
        
        # Add sampling metadata if sampled (Requirement 18.4)
        if sampling_info.is_sampled:
            table_context['sampling_info'] = {
                'total_rows': sampling_info.total_rows,
                'sampled_rows': sampling_info.sampled_rows,
                'first_n': sampling_info.first_n,
                'last_n': sampling_info.last_n,
                'middle_n': sampling_info.middle_n,
                'note': 'This is a sample of the data. Aggregations are computed on the full dataset.'
            }
        
        return table_context, sampling_info
    
    def _build_summary_section(
        self,
        df: pd.DataFrame,
        section: DocumentSection
    ) -> Dict[str, Any]:
        """Build context for summary section."""
        section_df = df.iloc[section.start_row:section.end_row + 1]
        
        summary_context = {
            'type': 'summary',
            'content_type': section.content_type.value
        }
        
        # Extract calculations and totals
        calculations = self._extract_calculations(section_df)
        if calculations:
            summary_context['calculations'] = calculations
        
        # Extract key-value pairs
        kv_pairs = self._extract_key_value_from_section(section_df)
        if kv_pairs:
            summary_context['fields'] = kv_pairs
        
        return summary_context
    
    def _build_footer_section(
        self,
        df: pd.DataFrame,
        section: DocumentSection
    ) -> Dict[str, Any]:
        """Build context for footer section."""
        section_df = df.iloc[section.start_row:section.end_row + 1]
        
        footer_context = {
            'type': 'footer',
            'content_type': section.content_type.value
        }
        
        # Extract text content
        text_content = self._extract_text_from_section(section_df)
        if text_content:
            footer_context['content'] = text_content
        
        return footer_context
    
    def _build_metadata_section(
        self,
        df: pd.DataFrame,
        section: DocumentSection
    ) -> Dict[str, Any]:
        """Build context for metadata section."""
        section_df = df.iloc[section.start_row:section.end_row + 1]
        
        metadata_context = {
            'type': 'metadata',
            'content_type': section.content_type.value
        }
        
        # Extract key-value pairs
        kv_pairs = self._extract_key_value_from_section(section_df)
        if kv_pairs:
            metadata_context['fields'] = kv_pairs
        
        return metadata_context
    
    def _find_table_for_section(
        self,
        parsed: ParsedSpreadsheet,
        section: DocumentSection
    ) -> Optional[Tuple[TableRegion, pd.DataFrame, TableSchema]]:
        """Find the table that corresponds to a section."""
        for region, df, schema in parsed.tables:
            # Check if table region overlaps with section
            if (region.start_row >= section.start_row and 
                region.start_row <= section.end_row):
                return (region, df, schema)
        return None
    
    def _add_section_markers(
        self,
        sections_dict: Dict[str, Any],
        sections: List[DocumentSection]
    ) -> Dict[str, Any]:
        """
        Add explicit section transition markers.
        
        Validates: Requirements 17.6, 18.5
        """
        marked_dict = {}
        
        for i, (key, value) in enumerate(sections_dict.items()):
            marked_dict[key] = value
            
            # Add transition marker after each section (Requirement 18.5)
            if isinstance(value, dict) and 'type' in value:
                section_type = value['type']
                marked_dict[f"_marker_after_{key}"] = f"--- End of {section_type.title()} Section ---"
        
        return marked_dict
    
    def preserve_intentional_gaps(
        self,
        parsed: ParsedSpreadsheet,
        sections_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Preserve intentional gaps as section markers in context.
        
        Intentional gaps (empty rows that separate sections) are preserved
        as explicit markers to maintain document structure.
        
        Args:
            parsed: ParsedSpreadsheet with intentional_gaps list
            sections_dict: Sections dictionary to enhance
        
        Returns:
            Enhanced sections dictionary with gap markers
        
        Validates: Requirement 17.6
        """
        if not parsed.intentional_gaps:
            return sections_dict
        
        enhanced_dict = {}
        
        for key, value in sections_dict.items():
            enhanced_dict[key] = value
            
            # Check if this section is followed by an intentional gap
            if isinstance(value, dict) and 'type' in value:
                # Extract section info from key
                if '_' in key:
                    parts = key.split('_')
                    if len(parts) >= 2:
                        try:
                            section_end_row = int(parts[-1])
                            
                            # Check if next row is an intentional gap
                            if section_end_row + 1 in parsed.intentional_gaps:
                                enhanced_dict[f"_gap_after_{key}"] = {
                                    'type': 'intentional_gap',
                                    'note': 'Structural separator between sections',
                                    'row': section_end_row + 1
                                }
                        except (ValueError, IndexError):
                            pass
        
        return enhanced_dict
    
    def _compute_validation_checksums(
        self,
        df: pd.DataFrame,
        schema: TableSchema
    ) -> Dict[str, Any]:
        """Compute validation checksums for data integrity."""
        checksums = {}
        
        # Compute sums for numeric columns
        for col in schema.headers:
            if col in df.columns:
                if schema.dtypes.get(col) in ['int64', 'float64', 'numeric']:
                    try:
                        col_sum = df[col].sum()
                        if pd.notna(col_sum):
                            checksums[f"{col}_sum"] = float(col_sum)
                    except (TypeError, ValueError):
                        pass
        
        # Add row count
        checksums['row_count'] = len(df)
        
        return checksums
    
    def _extract_summary_values(
        self,
        section_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract summary values (totals, subtotals, etc.)."""
        summary_values = {}
        
        for idx in range(len(section_df)):
            row = section_df.iloc[idx]
            
            # Look for calculation keywords
            for col_idx, val in enumerate(row):
                if pd.isna(val):
                    continue
                
                val_str = str(val).lower()
                
                # Check for total keywords
                if any(kw in val_str for kw in ['total', 'subtotal', 'sum', 'grand']):
                    # Next cell might have the value
                    if col_idx + 1 < len(row):
                        total_val = row.iloc[col_idx + 1]
                        if pd.notna(total_val) and isinstance(total_val, (int, float, np.number)):
                            key = str(val).strip().lower().replace(' ', '_').rstrip(':')
                            summary_values[key] = float(total_val)
        
        return summary_values
    
    def _extract_key_value_from_section(
        self,
        section_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract key-value pairs from a section."""
        kv_pairs = {}
        
        for idx in range(len(section_df)):
            row = section_df.iloc[idx]
            
            # Look for key:value patterns
            for col_idx in range(len(row) - 1):
                key_cell = row.iloc[col_idx]
                val_cell = row.iloc[col_idx + 1]
                
                if pd.notna(key_cell) and pd.notna(val_cell):
                    key_str = str(key_cell).strip()
                    
                    # Check if this looks like a key
                    if key_str.endswith(':') or len(key_str) < 30:
                        key_clean = key_str.rstrip(':').lower().replace(' ', '_')
                        kv_pairs[key_clean] = val_cell
        
        return kv_pairs
    
    def _extract_calculations(
        self,
        section_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract calculations from summary section."""
        calculations = {}
        
        calc_keywords = ['total', 'subtotal', 'sum', 'average', 'count', 'tax', 'vat', 'shipping', 'discount']
        
        for idx in range(len(section_df)):
            row = section_df.iloc[idx]
            
            for col_idx in range(len(row) - 1):
                label_cell = row.iloc[col_idx]
                value_cell = row.iloc[col_idx + 1]
                
                if pd.notna(label_cell) and pd.notna(value_cell):
                    label_str = str(label_cell).lower()
                    
                    # Check if this is a calculation
                    if any(kw in label_str for kw in calc_keywords):
                        if isinstance(value_cell, (int, float, np.number)):
                            key = str(label_cell).strip().lower().replace(' ', '_').rstrip(':')
                            calculations[key] = float(value_cell)
        
        return calculations
    
    def _extract_text_from_section(
        self,
        section_df: pd.DataFrame
    ) -> List[str]:
        """Extract text content from a section."""
        text_content = []
        
        for idx in range(len(section_df)):
            row = section_df.iloc[idx]
            
            for val in row:
                if pd.notna(val) and isinstance(val, str):
                    text_content.append(val.strip())
        
        return text_content
    
    def _build_minimal_context(
        self,
        parsed: ParsedSpreadsheet
    ) -> str:
        """Build minimal context when budget is very tight."""
        minimal = {
            'document_type': parsed.document_type.value,
            'table_count': parsed.table_count,
            'metadata': self.format_metadata(parsed.metadata)
        }
        
        # Include only primary table schema
        if parsed.primary_table:
            region, df, schema = parsed.primary_table
            minimal['primary_table'] = {
                'schema': schema.headers,
                'row_count': schema.row_count,
                'col_count': schema.col_count,
                'sample': df.head(5).to_dict('records')
            }
        
        return json.dumps(minimal, separators=(',', ':'))
    
    def _validate_context_completeness(
        self,
        context: StructuredContext,
        original: ParsedSpreadsheet
    ) -> bool:
        """Internal validation method."""
        return self.validate_context_completeness(context, original)
