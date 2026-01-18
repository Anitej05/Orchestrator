"""
SpreadsheetParser - Main orchestrator class for intelligent spreadsheet parsing.

This class coordinates all parsing components to provide comprehensive
spreadsheet analysis including table detection, schema extraction,
metadata extraction, and context building.

Enhanced with advanced performance optimizations.

Requirements: 1, 2, 4, 16, 17, 18
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd

from .parsing_models import (
    ParsedSpreadsheet,
    DocumentType,
    DocumentSection,
    SectionType,
    ContentType,
    TableRegion,
    TableSchema,
    SamplingStrategy,
    StructuredContext,
    create_table_region_from_df,
    create_table_schema_from_df
)

from .parsing import (
    DocumentSectionDetector,
    IntentionalGapDetector,
    TableDetector,
    ContextBuilder,
    MetadataExtractor,
    SchemaExtractor
)

logger = logging.getLogger(__name__)

# Import performance optimizations
try:
    from .performance_optimizer import (
        performance_monitor,
        memory_optimizer,
        token_optimizer,
        advanced_cache
    )
    PERFORMANCE_OPTIMIZATION_ENABLED = True
    logger.info("✅ Performance optimizations enabled for SpreadsheetParser")
except ImportError as e:
    logger.warning(f"⚠️ Performance optimizations not available: {e}")
    PERFORMANCE_OPTIMIZATION_ENABLED = False


class SpreadsheetParser:
    """
    Main orchestrator class for intelligent spreadsheet parsing.
    
    Coordinates all parsing components to provide comprehensive analysis:
    - Document structure detection
    - Table boundary identification
    - Schema extraction with type inference
    - Metadata extraction
    - Context building for LLM consumption
    - Intentional gap detection
    
    Requirements: 1.1-1.5, 2.1-2.5, 4.1-4.4, 16.1-16.7, 17.1-17.6, 18.1-18.7
    """
    
    def __init__(self):
        """Initialize the spreadsheet parser with all components."""
        self.document_section_detector = DocumentSectionDetector()
        self.intentional_gap_detector = IntentionalGapDetector()
        self.table_detector = TableDetector()
        self.context_builder = ContextBuilder()
        self.metadata_extractor = MetadataExtractor()
        self.schema_extractor = SchemaExtractor()
        
        self.logger = logging.getLogger(f"{__name__}.SpreadsheetParser")
        
        # Performance tracking
        self._parse_times = []
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance optimization integration
        if PERFORMANCE_OPTIMIZATION_ENABLED:
            self.logger.info("🚀 Advanced performance monitoring enabled")
        else:
            self.logger.info("📊 Basic performance tracking enabled")
    
    def parse_file(
        self,
        file_path: str,
        file_id: str,
        sheet_name: Optional[str] = None,
        max_rows: Optional[int] = None,
        parse_all_sheets: bool = False
    ) -> ParsedSpreadsheet:
        """
        Parse a spreadsheet file with intelligent structure detection.
        
        Args:
            file_path: Path to the spreadsheet file
            file_id: Unique identifier for the file
            sheet_name: Specific sheet to parse (None for first sheet)
            max_rows: Maximum rows to process (None for all rows)
            parse_all_sheets: If True, parse all sheets and return summary
            
        Returns:
            ParsedSpreadsheet with complete analysis
            
        Requirements: 1.1, 1.2, 5.1, 5.2, 16.1, 16.5
        """
        # Performance monitoring
        if PERFORMANCE_OPTIMIZATION_ENABLED:
            timer = performance_monitor.time_operation("parse_file")
            timer.__enter__()
        
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Parsing file: {file_path} (sheet: {sheet_name}, all_sheets: {parse_all_sheets})")
            
            # Check cache first
            cache_key = f"parse:{file_id}:{sheet_name}:{max_rows}:{parse_all_sheets}"
            if PERFORMANCE_OPTIMIZATION_ENABLED:
                cached_result = advanced_cache.get(cache_key)
                if cached_result:
                    self._cache_hits += 1
                    self.logger.info(f"🎯 Cache HIT for {file_id}")
                    return cached_result
                self._cache_misses += 1
            
            # Handle multi-sheet parsing
            if parse_all_sheets and Path(file_path).suffix.lower() in ['.xlsx', '.xls']:
                result = self._parse_all_sheets(file_path, file_id, max_rows)
            else:
                # Load the spreadsheet
                raw_df = self._load_spreadsheet(file_path, sheet_name, max_rows)
                
                if raw_df is None or raw_df.empty:
                    self.logger.warning(f"Empty or invalid spreadsheet: {file_path}")
                    result = self._create_empty_parsed_spreadsheet(file_id, sheet_name or "Sheet1")
                else:
                    self.logger.info(f"📊 Loaded spreadsheet: {len(raw_df)} rows × {len(raw_df.columns)} cols")
                    
                    # Track memory usage
                    if PERFORMANCE_OPTIMIZATION_ENABLED:
                        memory_usage = raw_df.memory_usage(deep=True).sum() / (1024 * 1024)
                        memory_optimizer.track_session_memory(file_id, memory_usage)
                    
                    # Detect document type
                    document_type = self._detect_document_type(raw_df)
                    self.logger.info(f"📋 Document type detected: {document_type.value}")
                    
                    # Detect document sections
                    sections = self.document_section_detector.detect_sections(raw_df)
                    self.logger.info(f"📑 Detected {len(sections)} document sections")
                    
                    # Detect intentional gaps
                    intentional_gaps = self.intentional_gap_detector.classify_empty_rows(raw_df)
                    self.logger.info(f"🔍 Classified {len(intentional_gaps)} empty rows")
                    
                    # Detect tables
                    table_regions = self.table_detector.detect_all_tables(raw_df)
                    self.logger.info(f"📊 Detected {len(table_regions)} table regions")
                    
                    # Process each table
                    tables = []
                    for region in table_regions:
                        table_df = self._extract_table_dataframe(raw_df, region)
                        schema = self.schema_extractor.extract_schema(table_df, region)
                        tables.append((region, table_df, schema))
                    
                    # Extract metadata
                    metadata = self.metadata_extractor.extract_metadata(raw_df, sections)
                    self.logger.info(f"📝 Extracted metadata: {len(metadata)} items")
                    
                    # Calculate parsing confidence
                    parsing_confidence = self._calculate_parsing_confidence(
                        sections, table_regions, metadata
                    )
                    
                    # Create parsed spreadsheet
                    result = ParsedSpreadsheet(
                        file_id=file_id,
                        sheet_name=sheet_name or "Sheet1",
                        document_type=document_type,
                        metadata=metadata,
                        sections=sections,
                        tables=tables,
                        raw_df=raw_df,
                        intentional_gaps=intentional_gaps,
                        parsing_confidence=parsing_confidence
                    )
            
            parse_time = time.time() - start_time
            self._parse_times.append(parse_time)
            
            # Cache the result
            if PERFORMANCE_OPTIMIZATION_ENABLED:
                estimated_size = len(str(result)) / (1024 * 1024)  # Rough estimate
                advanced_cache.put(cache_key, result, estimated_size)
            
            self.logger.info(
                f"✅ Parsing complete in {parse_time:.2f}s "
                f"(confidence: {getattr(result, 'parsing_confidence', 0.0):.2f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Parsing failed for {file_path}: {e}", exc_info=True)
            # Return basic parsed spreadsheet with error info
            result = self._create_error_parsed_spreadsheet(file_id, sheet_name, str(e))
            return result
        finally:
            if PERFORMANCE_OPTIMIZATION_ENABLED and 'timer' in locals():
                timer.__exit__(None, None, None)
    
    def parse_dataframe(
        self,
        df: pd.DataFrame,
        file_id: str,
        sheet_name: str = "Sheet1"
    ) -> ParsedSpreadsheet:
        """
        Parse an already-loaded DataFrame with intelligent structure detection.
        
        Args:
            df: The pandas DataFrame to parse
            file_id: Unique identifier for the file
            sheet_name: Name of the sheet
            
        Returns:
            ParsedSpreadsheet with complete analysis
            
        Requirements: 1.1, 1.2, 16.1
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Parsing DataFrame: {len(df)} rows × {len(df.columns)} cols")
            
            if df.empty:
                return self._create_empty_parsed_spreadsheet(file_id, sheet_name)
            
            # Detect document type
            document_type = self._detect_document_type(df)
            
            # Detect document sections
            sections = self.document_section_detector.detect_sections(df)
            
            # Detect intentional gaps
            intentional_gaps = self.intentional_gap_detector.classify_empty_rows(df)
            
            # Detect tables
            table_regions = self.table_detector.detect_all_tables(df)
            
            # Process each table
            tables = []
            for region in table_regions:
                table_df = self._extract_table_dataframe(df, region)
                schema = self.schema_extractor.extract_schema(table_df, region)
                tables.append((region, table_df, schema))
            
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(df, sections)
            
            # Calculate parsing confidence
            parsing_confidence = self._calculate_parsing_confidence(
                sections, table_regions, metadata
            )
            
            # Create parsed spreadsheet
            parsed = ParsedSpreadsheet(
                file_id=file_id,
                sheet_name=sheet_name,
                document_type=document_type,
                metadata=metadata,
                sections=sections,
                tables=tables,
                raw_df=df,
                intentional_gaps=intentional_gaps,
                parsing_confidence=parsing_confidence
            )
            
            parse_time = time.time() - start_time
            self._parse_times.append(parse_time)
            
            self.logger.info(
                f"✅ DataFrame parsing complete in {parse_time:.2f}s "
                f"(confidence: {parsing_confidence:.2f})"
            )
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"❌ DataFrame parsing failed: {e}", exc_info=True)
            return self._create_error_parsed_spreadsheet(file_id, sheet_name, str(e))
    
    def build_context(
        self,
        parsed: ParsedSpreadsheet,
        max_tokens: int = 8000,
        sampling_strategy: Optional[SamplingStrategy] = None
    ) -> StructuredContext:
        """
        Build structured context for LLM consumption with token optimization.
        
        Args:
            parsed: The parsed spreadsheet
            max_tokens: Maximum tokens for the context
            sampling_strategy: Strategy for sampling large datasets (currently unused)
            
        Returns:
            StructuredContext ready for LLM consumption
            
        Requirements: 3.1, 3.2, 8.1, 8.3, 18.1-18.7
        """
        try:
            # Use token optimizer if available
            if PERFORMANCE_OPTIMIZATION_ENABLED:
                with performance_monitor.time_operation("build_context"):
                    # Get primary table for optimization
                    primary_table = self.get_primary_table(parsed)
                    if primary_table:
                        region, table_df, schema = primary_table
                        
                        # Use token optimizer for efficient context building
                        optimized_context = token_optimizer.optimize_dataframe_context(
                            df=table_df,
                            max_tokens=max_tokens,
                            include_columns=None,  # Include all columns
                            priority_columns=None  # No priority columns for now
                        )
                        
                        # Convert to StructuredContext format
                        return StructuredContext(
                            document_type=parsed.document_type,
                            sections={
                                "optimized_table": {
                                    "type": "table",
                                    "schema": optimized_context["schema"],
                                    "sample_data": optimized_context["sample_data"],
                                    "metadata": optimized_context["metadata"]
                                },
                                "document_metadata": {
                                    "type": "metadata",
                                    "content": parsed.metadata,
                                    "sections_count": len(parsed.sections),
                                    "parsing_confidence": parsed.parsing_confidence
                                }
                            }
                        )
            
            # Fallback to standard context building
            return self.context_builder.build_structured_context(
                parsed, max_tokens
            )
        except Exception as e:
            self.logger.error(f"❌ Context building failed: {e}", exc_info=True)
            # Return basic context
            return StructuredContext(
                document_type=parsed.document_type,
                sections={
                    "error": {
                        "type": "error",
                        "message": f"Context building failed: {str(e)}"
                    }
                }
            )
    
    def get_primary_table(self, parsed: ParsedSpreadsheet) -> Optional[Tuple[TableRegion, pd.DataFrame, TableSchema]]:
        """
        Get the primary (largest) table from parsed spreadsheet.
        
        Args:
            parsed: The parsed spreadsheet
            
        Returns:
            Tuple of (TableRegion, DataFrame, TableSchema) or None
            
        Requirements: 1.2
        """
        return parsed.primary_table
    
    def get_metadata_summary(self, parsed: ParsedSpreadsheet) -> Dict[str, Any]:
        """
        Get a summary of extracted metadata.
        
        Args:
            parsed: The parsed spreadsheet
            
        Returns:
            Dictionary with metadata summary
            
        Requirements: 4.1, 4.4
        """
        return {
            "document_type": parsed.document_type.value,
            "has_metadata": parsed.has_metadata(),
            "has_line_items": parsed.has_line_items(),
            "has_summary": parsed.has_summary(),
            "sections_count": len(parsed.sections),
            "tables_count": len(parsed.tables),
            "intentional_gaps": len(parsed.intentional_gaps),
            "parsing_confidence": parsed.parsing_confidence,
            "metadata_items": len(parsed.metadata),
            "metadata_keys": list(parsed.metadata.keys()) if parsed.metadata else []
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the parser."""
        if not self._parse_times:
            return {"no_data": True}
        
        return {
            "total_parses": len(self._parse_times),
            "avg_parse_time": sum(self._parse_times) / len(self._parse_times),
            "min_parse_time": min(self._parse_times),
            "max_parse_time": max(self._parse_times),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }
    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    def _load_spreadsheet(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        max_rows: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Load spreadsheet file into DataFrame with advanced edge case handling."""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path, nrows=max_rows)
            elif file_ext in ['.xlsx', '.xls']:
                # Use advanced edge case handler for Excel files
                from .edge_case_handler import edge_case_handler
                
                try:
                    # Try advanced processing first
                    df = edge_case_handler.process_excel_file(
                        file_path=file_path,
                        sheet_name=sheet_name,
                        handle_merged_cells=True,
                        extract_formulas=True,
                        handle_errors=True
                    )
                    
                    # Apply row limit if specified
                    if max_rows and len(df) > max_rows:
                        df = df.head(max_rows)
                    
                    self.logger.info(f"✅ Advanced Excel processing successful: {len(df)} rows × {len(df.columns)} cols")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Advanced Excel processing failed: {e}, falling back to basic pandas")
                    # Fallback to basic pandas loading
                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet_name or 0,
                        nrows=max_rows
                    )
            else:
                self.logger.error(f"Unsupported file format: {file_ext}")
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load spreadsheet {file_path}: {e}")
            return None
    
    def _detect_document_type(self, df: pd.DataFrame) -> DocumentType:
        """Detect the type of document based on content patterns."""
        try:
            # Simple heuristics for document type detection
            # This could be enhanced with ML models in the future
            
            # Check for invoice patterns
            text_content = ' '.join([
                str(cell).lower() for cell in df.iloc[:10].values.flatten()
                if pd.notna(cell) and isinstance(cell, (str, int, float))
            ])
            
            if any(keyword in text_content for keyword in ['invoice', 'bill', 'total', 'amount', 'due']):
                return DocumentType.INVOICE
            
            if any(keyword in text_content for keyword in ['report', 'summary', 'analysis']):
                return DocumentType.REPORT
            
            if any(keyword in text_content for keyword in ['form', 'application', 'request']):
                return DocumentType.FORM
            
            # Default to data table if it looks structured
            if len(df.columns) > 2 and len(df) > 5:
                return DocumentType.DATA_TABLE
            
            return DocumentType.UNKNOWN
            
        except Exception as e:
            self.logger.warning(f"Document type detection failed: {e}")
            return DocumentType.UNKNOWN
    
    def _extract_table_dataframe(self, df: pd.DataFrame, region: TableRegion) -> pd.DataFrame:
        """Extract a DataFrame for a specific table region."""
        try:
            return df.iloc[
                region.start_row:region.end_row + 1,
                region.start_col:region.end_col + 1
            ].copy()
        except Exception as e:
            self.logger.error(f"Failed to extract table DataFrame: {e}")
            return pd.DataFrame()
    
    def _calculate_parsing_confidence(
        self,
        sections: List[DocumentSection],
        table_regions: List[TableRegion],
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate overall parsing confidence score."""
        try:
            confidence_factors = []
            
            # Section detection confidence
            if sections:
                section_confidence = min(1.0, len(sections) / 3)  # Expect ~3 sections
                confidence_factors.append(section_confidence)
            
            # Table detection confidence
            if table_regions:
                table_confidence = sum(region.confidence for region in table_regions) / len(table_regions)
                confidence_factors.append(table_confidence)
            
            # Metadata extraction confidence
            if metadata:
                metadata_confidence = min(1.0, len(metadata) / 5)  # Expect ~5 metadata items
                confidence_factors.append(metadata_confidence)
            
            # Overall confidence is the average of all factors
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5  # Neutral confidence if no factors
                
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _create_empty_parsed_spreadsheet(self, file_id: str, sheet_name: str) -> ParsedSpreadsheet:
        """Create a ParsedSpreadsheet for empty files."""
        return ParsedSpreadsheet(
            file_id=file_id,
            sheet_name=sheet_name,
            document_type=DocumentType.UNKNOWN,
            metadata={"error": "Empty spreadsheet"},
            sections=[],
            tables=[],
            raw_df=pd.DataFrame(),
            intentional_gaps=[],
            parsing_confidence=0.0
        )
    
    def _create_error_parsed_spreadsheet(
        self,
        file_id: str,
        sheet_name: Optional[str],
        error_message: str
    ) -> ParsedSpreadsheet:
        """Create a ParsedSpreadsheet for parsing errors."""
        return ParsedSpreadsheet(
            file_id=file_id,
            sheet_name=sheet_name or "Unknown",
            document_type=DocumentType.UNKNOWN,
            metadata={"error": error_message},
            sections=[],
            tables=[],
            raw_df=pd.DataFrame(),
            intentional_gaps=[],
            parsing_confidence=0.0
        )
    
    def _parse_all_sheets(
        self,
        file_path: str,
        file_id: str,
        max_rows: Optional[int] = None
    ) -> ParsedSpreadsheet:
        """
        Parse all sheets in a workbook and return a summary.
        
        Requirements: 5.1, 5.2
        """
        try:
            # Get all sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            self.logger.info(f"📚 Found {len(sheet_names)} sheets: {sheet_names}")
            
            # Parse each sheet
            sheet_summaries = []
            all_sections = []
            all_tables = []
            combined_metadata = {"sheets": {}}
            
            for sheet_name in sheet_names:
                try:
                    # Parse individual sheet
                    sheet_parsed = self.parse_dataframe(
                        excel_file.parse(sheet_name, nrows=max_rows),
                        f"{file_id}_{sheet_name}",
                        sheet_name
                    )
                    
                    # Add to summary
                    sheet_summaries.append({
                        "name": sheet_name,
                        "rows": sheet_parsed.total_rows,
                        "columns": sheet_parsed.total_cols,
                        "document_type": sheet_parsed.document_type.value,
                        "sections": len(sheet_parsed.sections),
                        "tables": len(sheet_parsed.tables),
                        "confidence": sheet_parsed.parsing_confidence
                    })
                    
                    # Combine sections and tables
                    all_sections.extend(sheet_parsed.sections)
                    all_tables.extend(sheet_parsed.tables)
                    
                    # Combine metadata
                    combined_metadata["sheets"][sheet_name] = sheet_parsed.metadata
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse sheet {sheet_name}: {e}")
                    sheet_summaries.append({
                        "name": sheet_name,
                        "error": str(e),
                        "rows": 0,
                        "columns": 0
                    })
            
            # Create combined metadata
            combined_metadata.update({
                "total_sheets": len(sheet_names),
                "sheet_names": sheet_names,
                "sheet_summaries": sheet_summaries,
                "multi_sheet_workbook": True
            })
            
            # Determine overall document type (most common)
            doc_types = [s.get("document_type", "unknown") for s in sheet_summaries if "document_type" in s]
            if doc_types:
                most_common_type = max(set(doc_types), key=doc_types.count)
                document_type = DocumentType(most_common_type)
            else:
                document_type = DocumentType.UNKNOWN
            
            # Calculate overall confidence
            confidences = [s.get("confidence", 0) for s in sheet_summaries if "confidence" in s]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Create summary parsed spreadsheet
            return ParsedSpreadsheet(
                file_id=file_id,
                sheet_name="Multi-Sheet Summary",
                document_type=document_type,
                metadata=combined_metadata,
                sections=all_sections,
                tables=all_tables,
                raw_df=pd.DataFrame(),  # No single raw_df for multi-sheet
                intentional_gaps=[],
                parsing_confidence=overall_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Multi-sheet parsing failed: {e}")
            return self._create_error_parsed_spreadsheet(file_id, "Multi-Sheet", str(e))


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create a global parser instance for use across the agent
spreadsheet_parser = SpreadsheetParser()