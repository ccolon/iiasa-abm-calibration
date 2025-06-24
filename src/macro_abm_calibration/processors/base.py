"""
Base classes for data processors.

This module defines the common interface and functionality for all data processors,
ensuring consistency and enabling pipeline composition.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

from ..models import Country, Industry, TimeFrame
from ..utils.logging import get_logger
from ..utils.validation import ValidationResult

logger = get_logger(__name__)


class ProcessingStatus(str, Enum):
    """Status of a processing operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingMetadata:
    """Metadata for processing operations."""
    processor_name: str
    operation_id: str
    input_datasets: List[str] = field(default_factory=list)
    output_datasets: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: Optional[float] = None
    rows_processed: Optional[int] = None
    error_message: Optional[str] = None


class ProcessingResult(BaseModel):
    """
    Result of a data processing operation.
    
    Attributes:
        data: Processed data as DataFrame or dictionary of DataFrames
        metadata: Processing metadata
        status: Processing status
        validation: Data validation results
    """
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    metadata: ProcessingMetadata
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    validation: Optional[ValidationResult] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda df: {"shape": df.shape, "columns": list(df.columns)},
            datetime: lambda dt: dt.isoformat(),
        }
    
    @property
    def is_success(self) -> bool:
        """Check if processing was successful."""
        return self.status == ProcessingStatus.COMPLETED
    
    @property
    def has_validation_errors(self) -> bool:
        """Check if there are validation errors."""
        return self.validation is not None and not self.validation.is_valid


class ProcessingError(Exception):
    """Base exception for processing errors."""
    
    def __init__(
        self, 
        message: str, 
        processor_name: Optional[str] = None, 
        operation_id: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize processing error."""
        super().__init__(message)
        self.processor_name = processor_name
        self.operation_id = operation_id
        self.original_error = original_error
        self.timestamp = datetime.now()


class DataProcessor(ABC):
    """
    Abstract base class for all data processors.
    
    This class defines the common interface that all data processors must implement,
    ensuring consistency and enabling pipeline composition.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize data processor.
        
        Args:
            name: Processor name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f"{__name__}.{self.name}")
        self._processing_history: List[ProcessingMetadata] = []
    
    @abstractmethod
    def process(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process data according to the processor's logic.
        
        Args:
            data: Input data to process
            parameters: Processing parameters
            
        Returns:
            ProcessingResult with processed data and metadata
        """
        pass
    
    def process_with_validation(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None,
        validate_input: bool = True,
        validate_output: bool = True
    ) -> ProcessingResult:
        """
        Process data with optional input/output validation.
        
        Args:
            data: Input data to process
            parameters: Processing parameters
            validate_input: Whether to validate input data
            validate_output: Whether to validate output data
            
        Returns:
            ProcessingResult with validation information
        """
        operation_id = self._generate_operation_id()
        start_time = time.time()
        
        metadata = ProcessingMetadata(
            processor_name=self.name,
            operation_id=operation_id,
            parameters=parameters or {}
        )
        
        try:
            self.logger.info(f"Starting processing operation {operation_id}")
            
            # Input validation
            if validate_input:
                input_validation = self.validate_input(data, parameters)
                if not input_validation.is_valid:
                    raise ProcessingError(
                        f"Input validation failed: {input_validation.errors}",
                        self.name,
                        operation_id
                    )
            
            # Process data
            result = self.process(data, parameters)
            
            # Update metadata
            execution_time = time.time() - start_time
            metadata.execution_time = execution_time
            
            if isinstance(result.data, pd.DataFrame):
                metadata.rows_processed = len(result.data)
            elif isinstance(result.data, dict):
                metadata.rows_processed = sum(len(df) for df in result.data.values())
            
            result.metadata = metadata
            
            # Output validation
            if validate_output:
                output_validation = self.validate_output(result.data, parameters)
                result.validation = output_validation
                
                if not output_validation.is_valid:
                    result.status = ProcessingStatus.FAILED
                    self.logger.warning(f"Output validation failed for {operation_id}")
            
            # Store in processing history
            self._processing_history.append(metadata)
            
            self.logger.info(
                f"Processing operation {operation_id} completed in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            metadata.execution_time = execution_time
            metadata.error_message = str(e)
            
            self.logger.error(f"Processing operation {operation_id} failed: {e}")
            
            raise ProcessingError(
                f"Processing failed: {e}",
                self.name,
                operation_id,
                e
            ) from e
    
    def validate_input(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            parameters: Processing parameters
            
        Returns:
            ValidationResult
        """
        # Default implementation - subclasses can override
        result = ValidationResult(is_valid=True)
        
        if isinstance(data, pd.DataFrame):
            if data.empty:
                result.add_error("Input DataFrame is empty")
        elif isinstance(data, dict):
            if not data:
                result.add_error("Input data dictionary is empty")
            
            for name, df in data.items():
                if not isinstance(df, pd.DataFrame):
                    result.add_error(f"Dataset '{name}' is not a DataFrame")
                elif df.empty:
                    result.add_warning(f"Dataset '{name}' is empty")
        else:
            result.add_error(f"Unsupported data type: {type(data)}")
        
        return result
    
    def validate_output(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate output data.
        
        Args:
            data: Output data to validate
            parameters: Processing parameters
            
        Returns:
            ValidationResult
        """
        # Default implementation - subclasses can override
        return self.validate_input(data, parameters)
    
    def get_processing_history(self) -> List[ProcessingMetadata]:
        """Get processing operation history."""
        return self._processing_history.copy()
    
    def clear_history(self) -> None:
        """Clear processing history."""
        self._processing_history.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get processor information."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "total_operations": len(self._processing_history),
            "successful_operations": sum(
                1 for m in self._processing_history if m.error_message is None
            ),
            "failed_operations": sum(
                1 for m in self._processing_history if m.error_message is not None
            )
        }
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{self.name}_{timestamp}"
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class ChainableProcessor(DataProcessor):
    """
    Base class for processors that can be chained together.
    
    This class adds functionality for combining processors into pipelines
    and handling data flow between processors.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize chainable processor."""
        super().__init__(name)
        self._next_processor: Optional[ChainableProcessor] = None
    
    def chain(self, next_processor: ChainableProcessor) -> ChainableProcessor:
        """
        Chain this processor with another processor.
        
        Args:
            next_processor: Processor to chain after this one
            
        Returns:
            The next processor (for fluent chaining)
        """
        self._next_processor = next_processor
        return next_processor
    
    def process_chain(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process data through the entire chain.
        
        Args:
            data: Input data
            parameters: Processing parameters
            
        Returns:
            Final processing result
        """
        # Process with this processor
        result = self.process_with_validation(data, parameters)
        
        if not result.is_success:
            return result
        
        # Continue with next processor if available
        if self._next_processor:
            return self._next_processor.process_chain(result.data, parameters)
        
        return result
    
    def get_chain_info(self) -> List[Dict[str, Any]]:
        """Get information about the entire processor chain."""
        chain_info = [self.get_info()]
        
        if self._next_processor:
            chain_info.extend(self._next_processor.get_chain_info())
        
        return chain_info


class BatchProcessor(DataProcessor):
    """
    Base class for processors that handle batch operations.
    
    This class provides functionality for processing multiple datasets
    or applying operations across multiple countries/time periods.
    """
    
    def __init__(self, name: Optional[str] = None, batch_size: Optional[int] = None):
        """
        Initialize batch processor.
        
        Args:
            name: Processor name
            batch_size: Maximum batch size for processing
        """
        super().__init__(name)
        self.batch_size = batch_size
    
    def process_batch(
        self,
        datasets: Dict[str, pd.DataFrame],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ProcessingResult]:
        """
        Process multiple datasets as a batch.
        
        Args:
            datasets: Dictionary of datasets to process
            parameters: Processing parameters
            
        Returns:
            Dictionary of processing results
        """
        results = {}
        
        # Process datasets in batches if batch_size is specified
        if self.batch_size:
            dataset_items = list(datasets.items())
            for i in range(0, len(dataset_items), self.batch_size):
                batch = dict(dataset_items[i:i + self.batch_size])
                batch_results = self._process_batch_chunk(batch, parameters)
                results.update(batch_results)
        else:
            results = self._process_batch_chunk(datasets, parameters)
        
        return results
    
    def _process_batch_chunk(
        self,
        batch: Dict[str, pd.DataFrame],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ProcessingResult]:
        """Process a chunk of the batch."""
        results = {}
        
        for name, data in batch.items():
            try:
                result = self.process_with_validation(data, parameters)
                results[name] = result
            except ProcessingError as e:
                # Create failed result
                metadata = ProcessingMetadata(
                    processor_name=self.name,
                    operation_id=self._generate_operation_id(),
                    error_message=str(e)
                )
                
                results[name] = ProcessingResult(
                    data=data,  # Return original data on failure
                    metadata=metadata,
                    status=ProcessingStatus.FAILED
                )
        
        return results