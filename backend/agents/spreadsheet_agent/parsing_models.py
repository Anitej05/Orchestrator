"""
Data models for intelligent spreadsheet parsing.

This module defines the core data structures used by the parsing system
to represent document structure, table regions, schemas, and metadata.

Requirements: 16.1, 16.2, 17.1, 18.1
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import pandas as pd


# ============================================================================
# ENUMS
# ============================================================================

class DocumentType(str, Enum):
    """Types of documents that can be detected"""
    INVOICE = "invoice"
    REPORT = "report"
    DATA_TABLE = "data_table"
    FORM = "form"
    UNKNOWN = "unknown"


class SectionType(str, Enum):
    """Types of document sections"""
    HEADER = "header"
    LINE_ITEMS = "line_items"
    SUMMARY = "summary"
    FOOTER = "footer"
    METADATA = "metadata"


class ContentType(str, Enum):
    """Types of content within sections"""
    TABLE = "table"
    KEY_VALUE = "key_value"
    TEXT = "text"
    CALCULATIONS = "calculations"


class GapType(str, Enum):
    """Types of gaps between rows"""
    STRUCTURAL_SEPARATOR = "structural_separator"
    SECTION_BOUNDARY = "section_boundary"
    MISSING_DATA = "missing_data"


# ============================================================================
# GAP DETECTION MODELS
# ============================================================================

@dataclass
class EmptyRowClassification:
    """
    Classification of an empty row within a spreadsheet.
    
    Requirements: 17.1, 17.2, 17.3, 17.4
    """
    row_index: int
    gap_type: GapType
    confidence: float
    context_before: str = ""
    context_after: str = ""
    reasoning: str = ""
    
    def __post_init__(self):
        """Validate classification data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    @property
    def is_structural(self) -> bool:
        """Check if this is a structural separator"""
        return self.gap_type in [GapType.STRUCTURAL_SEPARATOR, GapType.SECTION_BOUNDARY]
    
    @property
    def is_missing_data(self) -> bool:
        """Check if this represents missing data"""
        return self.gap_type == GapType.MISSING_DATA


# ============================================================================
# CORE DATA MODELS
# ============================================================================

@dataclass
class TableRegion:
    """
    Represents a rectangular data region within a spreadsheet.
    
    Requirements: 1.1, 1.2, 1.5
    """
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    confidence: float = 1.0
    header_row: Optional[int] = None
    
    def __post_init__(self):
        """Validate region bounds"""
        if self.start_row > self.end_row:
            raise ValueError("start_row cannot be greater than end_row")
        if self.start_col > self.end_col:
            raise ValueError("start_col cannot be greater than end_col")
    
    @property
    def row_count(self) -> int:
        """Number of rows in the region"""
        return self.end_row - self.start_row + 1
    
    @property
    def col_count(self) -> int:
        """Number of columns in the region"""
        return self.end_col - self.start_col + 1
    
    @property
    def size(self) -> int:
        """Total number of cells in the region"""
        return self.row_count * self.col_count
    
    def contains_row(self, row: int) -> bool:
        """Check if a row is within this region"""
        return self.start_row <= row <= self.end_row
    
    def contains_col(self, col: int) -> bool:
        """Check if a column is within this region"""
        return self.start_col <= col <= self.end_col
    
    def overlaps_with(self, other: 'TableRegion') -> bool:
        """Check if this region overlaps with another"""
        return not (
            self.end_row < other.start_row or
            self.start_row > other.end_row or
            self.end_col < other.start_col or
            self.start_col > other.end_col
        )


@dataclass
class DocumentSection:
    """
    Represents a distinct section within a document.
    
    Requirements: 16.1, 16.2, 17.1
    """
    section_type: SectionType
    start_row: int
    end_row: int
    content_type: ContentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate section bounds"""
        if self.start_row > self.end_row:
            raise ValueError("start_row cannot be greater than end_row")
    
    @property
    def row_count(self) -> int:
        """Number of rows in the section"""
        return self.end_row - self.start_row + 1
    
    def contains_row(self, row: int) -> bool:
        """Check if a row is within this section"""
        return self.start_row <= row <= self.end_row


@dataclass
class TableSchema:
    """
    Schema information for a data table.
    
    Requirements: 2.1, 2.2, 2.4, 2.5
    """
    headers: List[str]
    dtypes: Dict[str, str]
    row_count: int
    col_count: int
    null_counts: Dict[str, int] = field(default_factory=dict)
    unique_counts: Dict[str, int] = field(default_factory=dict)
    sample_values: Dict[str, List[Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate schema consistency"""
        if len(self.headers) != self.col_count:
            raise ValueError("Number of headers must match col_count")
        if len(self.dtypes) != self.col_count:
            raise ValueError("Number of dtypes must match col_count")
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names"""
        numeric_types = ['int64', 'float64', 'int32', 'float32', 'number']
        return [col for col, dtype in self.dtypes.items() if dtype in numeric_types]
    
    def get_text_columns(self) -> List[str]:
        """Get list of text column names"""
        text_types = ['object', 'string', 'text']
        return [col for col, dtype in self.dtypes.items() if dtype in text_types]
    
    def get_date_columns(self) -> List[str]:
        """Get list of date column names"""
        date_types = ['datetime64', 'date', 'datetime']
        return [col for col, dtype in self.dtypes.items() if any(dt in dtype for dt in date_types)]


@dataclass
class SamplingStrategy:
    """
    Strategy for sampling large datasets.
    
    Requirements: 3.1, 3.2, 8.1
    """
    strategy_name: str  # 'all_rows', 'head_tail', 'stratified', 'smart'
    total_rows: int
    sampled_rows: int
    first_n: int = 0
    last_n: int = 0
    middle_n: int = 0
    include_totals: bool = True
    
    def __post_init__(self):
        """Validate strategy parameters"""
        valid_names = ['all_rows', 'head_tail', 'stratified', 'smart']
        if self.strategy_name not in valid_names:
            raise ValueError(f"Invalid strategy name: {self.strategy_name}")
        
        if self.sampled_rows > self.total_rows:
            raise ValueError("Sampled rows cannot exceed total rows")
    
    @property
    def sampling_ratio(self) -> float:
        """Calculate the sampling ratio"""
        if self.total_rows == 0:
            return 0.0
        return self.sampled_rows / self.total_rows
    
    @property
    def is_sampled(self) -> bool:
        """Check if sampling was applied"""
        return self.sampled_rows < self.total_rows


@dataclass
class ParsedSpreadsheet:
    """
    Complete representation of a parsed spreadsheet with intelligent structure detection.
    
    Requirements: 16.1, 16.2, 17.1, 18.1
    """
    file_id: str
    sheet_name: str
    document_type: DocumentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[DocumentSection] = field(default_factory=list)
    tables: List[Tuple[TableRegion, pd.DataFrame, TableSchema]] = field(default_factory=list)
    raw_df: Optional[pd.DataFrame] = None
    intentional_gaps: List[int] = field(default_factory=list)
    parsing_confidence: float = 1.0
    
    @property
    def primary_table(self) -> Optional[Tuple[TableRegion, pd.DataFrame, TableSchema]]:
        """Get the primary (largest) table"""
        if not self.tables:
            return None
        return max(self.tables, key=lambda t: t[0].size)
    
    @property
    def total_rows(self) -> int:
        """Total number of rows in the raw dataframe"""
        return len(self.raw_df) if self.raw_df is not None else 0
    
    @property
    def total_cols(self) -> int:
        """Total number of columns in the raw dataframe"""
        return len(self.raw_df.columns) if self.raw_df is not None else 0
    
    def get_section_by_type(self, section_type: SectionType) -> Optional[DocumentSection]:
        """Get the first section of a specific type"""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None
    
    def get_sections_by_type(self, section_type: SectionType) -> List[DocumentSection]:
        """Get all sections of a specific type"""
        return [section for section in self.sections if section.section_type == section_type]
    
    def has_metadata(self) -> bool:
        """Check if the document has metadata sections"""
        return any(section.section_type == SectionType.METADATA for section in self.sections)
    
    def has_line_items(self) -> bool:
        """Check if the document has line item sections"""
        return any(section.section_type == SectionType.LINE_ITEMS for section in self.sections)
    
    def has_summary(self) -> bool:
        """Check if the document has summary sections"""
        return any(section.section_type == SectionType.SUMMARY for section in self.sections)


# ============================================================================
# ANOMALY DETECTION MODELS
# ============================================================================

@dataclass
class DataAnomaly:
    """
    Represents a data quality issue detected during parsing.
    
    Requirements: 10.1, 10.2, 10.6, 10.7
    """
    type: str  # 'dtype_drift', 'missing_values', 'outliers', 'inconsistent_format'
    columns: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    sample_values: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggested_fixes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate anomaly data"""
        valid_types = ['dtype_drift', 'missing_values', 'outliers', 'inconsistent_format']
        if self.type not in valid_types:
            raise ValueError(f"Invalid anomaly type: {self.type}")
        
        valid_severities = ['low', 'medium', 'high', 'critical']
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity: {self.severity}")


# ============================================================================
# CONTEXT BUILDING MODELS
# ============================================================================

@dataclass
class ContextMetadata:
    """
    Metadata for LLM context generation.
    
    Requirements: 8.1, 8.4, 18.1
    """
    token_count: int = 0
    sampling_applied: bool = False
    sampling_strategy: Optional[SamplingStrategy] = None
    sections_included: List[str] = field(default_factory=list)
    anti_hallucination_markers: bool = True
    validation_checksums: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredContext:
    """
    Structured representation of spreadsheet content for LLM consumption.
    
    Requirements: 18.1, 18.2, 18.3, 18.4
    """
    document_type: DocumentType
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: ContextMetadata = field(default_factory=ContextMetadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "document_type": self.document_type.value,
            "sections": self.sections,
            "metadata": {
                "token_count": self.metadata.token_count,
                "sampling_applied": self.metadata.sampling_applied,
                "sections_included": self.metadata.sections_included,
                "anti_hallucination_markers": self.metadata.anti_hallucination_markers,
                "validation_checksums": self.metadata.validation_checksums
            }
        }
    
    def add_section(self, section_name: str, section_data: Dict[str, Any]) -> None:
        """Add a section to the structured context"""
        self.sections[section_name] = section_data
        if section_name not in self.metadata.sections_included:
            self.metadata.sections_included.append(section_name)
    
    def add_validation_checksum(self, key: str, value: Any) -> None:
        """Add a validation checksum to prevent hallucination"""
        self.metadata.validation_checksums[key] = value


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_table_region_from_df(df: pd.DataFrame, start_row: int = 0) -> TableRegion:
    """Create a TableRegion that covers the entire DataFrame"""
    return TableRegion(
        start_row=start_row,
        end_row=start_row + len(df) - 1,
        start_col=0,
        end_col=len(df.columns) - 1,
        confidence=1.0
    )


def create_table_schema_from_df(df: pd.DataFrame) -> TableSchema:
    """Create a TableSchema from a pandas DataFrame"""
    return TableSchema(
        headers=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        row_count=len(df),
        col_count=len(df.columns),
        null_counts=df.isnull().sum().to_dict(),
        unique_counts=df.nunique().to_dict(),
        sample_values={
            col: df[col].dropna().head(3).tolist() 
            for col in df.columns
        }
    )


def merge_table_regions(regions: List[TableRegion]) -> Optional[TableRegion]:
    """Merge multiple table regions into a single bounding region"""
    if not regions:
        return None
    
    min_start_row = min(r.start_row for r in regions)
    max_end_row = max(r.end_row for r in regions)
    min_start_col = min(r.start_col for r in regions)
    max_end_col = max(r.end_col for r in regions)
    
    # Average confidence
    avg_confidence = sum(r.confidence for r in regions) / len(regions)
    
    return TableRegion(
        start_row=min_start_row,
        end_row=max_end_row,
        start_col=min_start_col,
        end_col=max_end_col,
        confidence=avg_confidence
    )