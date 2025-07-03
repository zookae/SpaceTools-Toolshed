# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Variable handling engine for Toolshed.

This module provides a reusable variable handling system that can store variables
from tool outputs and resolve variable references in tool inputs using the $ syntax.
The resolution is recursive, handling nested dictionaries and lists.

Example usage:
    engine = VariableEngine()
    
    # Store variables from tool result
    result = ToolResult(image, variables={"processed_image": image})
    engine.store_variables(result.variables)
    
    # Resolve variables in tool arguments
    args = {"input_image": "$processed_image", "params": {"threshold": 0.5}}
    resolved_args = engine.resolve_variables(args)
"""

import re
import logging
from typing import Any, Dict, List, Union, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class VariableEngine:
    """
    Engine for storing and resolving variables in tool interactions.
    
    This class provides:
    - Variable storage from tool results
    - Recursive variable resolution using $ syntax
    - Support for nested data structures (dicts, lists)
    - Session-based variable scoping
    """
    
    def __init__(self):
        """Initialize the variable engine."""
        self.variables: Dict[str, Any] = {}
        self._variable_pattern = re.compile(r'\$([a-zA-Z_][a-zA-Z0-9_]*)')
    
    def store_variables(self, variables: Dict[str, Any]) -> None:
        """
        Store variables from a tool result.
        
        Args:
            variables: Dictionary of variable names and values to store
        """
        if not variables:
            return
            
        logger.debug("Storing %d variables: %s", len(variables), list(variables.keys()))
        self.variables.update(variables)
    
    def store_variable(self, name: str, value: Any) -> None:
        """
        Store a single variable.
        
        Args:
            name: Variable name
            value: Variable value
        """
        logger.debug("Storing variable '%s'", name)
        self.variables[name] = value
    
    def get_variable(self, name: str) -> Any:
        """
        Get a variable by name.
        
        Args:
            name: Variable name
            
        Returns:
            Variable value
            
        Raises:
            KeyError: If variable doesn't exist
        """
        if name not in self.variables:
            raise KeyError(f"Variable '${name}' not found. Available variables: {list(self.variables.keys())}")
        return self.variables[name]
    
    def has_variable(self, name: str) -> bool:
        """
        Check if a variable exists.
        
        Args:
            name: Variable name
            
        Returns:
            True if variable exists, False otherwise
        """
        return name in self.variables
    
    def list_variables(self) -> List[str]:
        """
        Get list of available variable names.
        
        Returns:
            List of variable names
        """
        return list(self.variables.keys())
    
    def clear_variables(self) -> None:
        """Clear all stored variables."""
        logger.debug("Clearing all variables")
        self.variables.clear()
    
    def resolve_variables(self, data: Any) -> Any:
        """
        Recursively resolve variable references in data structures.
        
        This method handles:
        - String variables: "$variable_name" -> variable value
        - Nested dictionaries: {"key": "$var"} -> {"key": resolved_value}
        - Nested lists: ["$var1", "$var2"] -> [value1, value2]
        - Mixed structures: {"images": ["$img1", "$img2"], "params": {"threshold": "$thresh"}}
        
        Args:
            data: Data structure that may contain variable references
            
        Returns:
            Data structure with variables resolved
            
        Raises:
            KeyError: If a referenced variable doesn't exist
        """
        return self._resolve_recursive(data)
    
    def _resolve_recursive(self, data: Any) -> Any:
        """
        Internal recursive resolver.
        
        Args:
            data: Data to resolve
            
        Returns:
            Resolved data
        """
        if isinstance(data, str):
            return self._resolve_string(data)
        elif isinstance(data, dict):
            return {key: self._resolve_recursive(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_recursive(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._resolve_recursive(item) for item in data)
        else:
            # Return unchanged for other types (int, float, bool, etc.)
            return data
    
    def _resolve_string(self, text: str) -> Any:
        """
        Resolve variables in a string.
        
        If the string is exactly "$variable_name", return the variable value directly.
        If the string contains variable references mixed with other text, substitute them.
        
        Args:
            text: String that may contain variable references
            
        Returns:
            Resolved value (may not be a string if the entire string was a variable reference)
        """
        # Check if the entire string is a single variable reference
        if text.startswith('$') and self._variable_pattern.fullmatch(text):
            var_name = text[1:]  # Remove the $ prefix
            return self.get_variable(var_name)
        
        # Handle mixed text with variable references
        def replace_var(match):
            var_name = match.group(1)
            try:
                var_value = self.get_variable(var_name)
                # Convert to string for substitution in mixed text
                return str(var_value)
            except KeyError:
                # Re-raise with more context
                raise KeyError(f"Variable '${var_name}' not found in string '{text}'. Available variables: {list(self.variables.keys())}")
        
        # Only substitute if there are variable references
        if '$' in text:
            return self._variable_pattern.sub(replace_var, text)
        else:
            return text
    
    def validate_references(self, data: Any) -> List[str]:
        """
        Validate that all variable references in data exist.
        
        Args:
            data: Data structure to validate
            
        Returns:
            List of missing variable names (empty if all valid)
        """
        missing_vars = []
        self._collect_missing_vars(data, missing_vars)
        return list(set(missing_vars))  # Remove duplicates
    
    def _collect_missing_vars(self, data: Any, missing_vars: List[str]) -> None:
        """
        Recursively collect missing variable references.
        
        Args:
            data: Data to check
            missing_vars: List to append missing variable names to
        """
        if isinstance(data, str):
            for match in self._variable_pattern.finditer(data):
                var_name = match.group(1)
                if not self.has_variable(var_name):
                    missing_vars.append(var_name)
        elif isinstance(data, dict):
            for value in data.values():
                self._collect_missing_vars(value, missing_vars)
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._collect_missing_vars(item, missing_vars)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored variables.
        
        Returns:
            Dictionary with variable statistics
        """
        return {
            "total_variables": len(self.variables),
            "variable_names": list(self.variables.keys()),
            "variable_types": {name: type(value).__name__ for name, value in self.variables.items()}
        }
    
    def export_variables(self) -> Dict[str, Any]:
        """
        Export all variables for serialization/backup.
        
        Returns:
            Deep copy of all variables
        """
        return deepcopy(self.variables)
    
    def import_variables(self, variables: Dict[str, Any], clear_existing: bool = False) -> None:
        """
        Import variables from external source.
        
        Args:
            variables: Variables to import
            clear_existing: Whether to clear existing variables first
        """
        if clear_existing:
            self.clear_variables()
        
        logger.debug("Importing %d variables", len(variables))
        self.variables.update(variables)


class SessionVariableEngine:
    """
    Session-aware variable engine for managing variables across multiple conversations.
    
    This extends the basic VariableEngine to support session-based variable scoping,
    useful for web applications and multi-user scenarios.
    """
    
    def __init__(self):
        """Initialize the session variable engine."""
        self.sessions: Dict[str, VariableEngine] = {}
    
    def get_session_engine(self, session_id: str) -> VariableEngine:
        """
        Get or create a variable engine for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            VariableEngine for the session
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = VariableEngine()
        return self.sessions[session_id]
    
    def store_variables(self, session_id: str, variables: Dict[str, Any]) -> None:
        """
        Store variables for a session.
        
        Args:
            session_id: Session identifier
            variables: Variables to store
        """
        engine = self.get_session_engine(session_id)
        engine.store_variables(variables)
    
    def resolve_variables(self, session_id: str, data: Any) -> Any:
        """
        Resolve variables for a session.
        
        Args:
            session_id: Session identifier
            data: Data with potential variable references
            
        Returns:
            Data with variables resolved
        """
        engine = self.get_session_engine(session_id)
        return engine.resolve_variables(data)
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all variables for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def list_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get variable statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session variable statistics
        """
        if session_id not in self.sessions:
            return {"total_variables": 0, "variable_names": [], "variable_types": {}}
        
        return self.sessions[session_id].get_stats()
