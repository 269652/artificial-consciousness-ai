"""
Advanced Logging System for ACI
Provides multi-level, multi-resolution logging with database persistence
"""
import os
import json
import logging
import logging.handlers
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import structlog
from rich.console import Console
from rich.logging import RichHandler

class LogLevel(Enum):
    """Log levels with different verbosity"""
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"

class LogResolution(Enum):
    """Different log resolutions for different use cases"""
    LEVEL1 = "level1"  # Top-level thoughts, actions, world events
    LEVEL2 = "level2"  # Component interactions, memory operations
    LEVEL3 = "level3"  # Detailed internal operations, API calls
    LEVEL4 = "level4"  # Full debug information, raw data

class ACILogger:
    """Advanced logging system for ACI components"""

    def __init__(self, log_dir: str = "logs", session_id: str = None):
        self.log_dir = Path(log_dir)
        self.base_log_dir = self.log_dir  # Add missing attribute
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = Console()

        # Initialize standard logging
        self._setup_standard_logging()

        # Initialize structured logging
        self._setup_structured_logging()

        # Create log files for different resolutions
        self._create_log_files()

    def _setup_standard_logging(self):
        """Setup standard Python logging with rich console output"""
        # Clear existing handlers
        logging.getLogger().handlers.clear()

        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )

        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s | %(pathname)s:%(lineno)d'
        )

        # Console handler with rich
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_level=True,
            show_path=False
        )
        console_handler.setLevel(logging.INFO)

        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)

    def _setup_structured_logging(self):
        """Setup structured logging with JSON output"""
        shared_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def _create_log_files(self):
        """Create rotating log files for different resolutions"""
        for resolution in LogResolution:
            # Create resolution-specific directory
            res_dir = self.log_dir / resolution.value
            res_dir.mkdir(exist_ok=True)

            # Create log files for each level
            for level in LogLevel:
                log_file = res_dir / f"{level.value.lower()}_{self.session_id}.log"

                # Rotating file handler
                handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=10*1024*1024, backupCount=5
                )
                handler.setLevel(self._get_logging_level(level))
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
                ))

                # Create logger for this resolution/level combination
                logger_name = f"aci.{resolution.value}.{level.value.lower()}"
                logger = logging.getLogger(logger_name)
                logger.setLevel(self._get_logging_level(level))
                logger.addHandler(handler)

    def _get_logging_level(self, level: LogLevel) -> int:
        """Convert LogLevel to Python logging level"""
        mapping = {
            LogLevel.ERROR: logging.ERROR,
            LogLevel.WARN: logging.WARNING,
            LogLevel.INFO: logging.INFO,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.TRACE: logging.DEBUG  # TRACE maps to DEBUG
        }
        return mapping[level]

    def log(self, level: LogLevel, resolution: LogResolution,
            component: str, message: str, **context):
        """Log a message with specified level and resolution"""
        logger_name = f"aci.{resolution.value}.{level.value.lower()}"
        logger = logging.getLogger(logger_name)

        # Add context information
        context.update({
            'component': component,
            'session_id': self.session_id,
            'resolution': resolution.value,
            'timestamp': datetime.now().isoformat()
        })

        # Log based on level
        if level == LogLevel.ERROR:
            logger.error(message, extra=context)
        elif level == LogLevel.WARN:
            logger.warning(message, extra=context)
        elif level == LogLevel.INFO:
            logger.info(message, extra=context)
        elif level == LogLevel.DEBUG or level == LogLevel.TRACE:
            logger.debug(message, extra=context)

        # Also log to database if available
        try:
            from .persistent_memory import get_memory_manager
            memory_manager = get_memory_manager()
            memory_manager.log_event(
                level=level.value,
                component=component,
                message=message,
                context=context,
                tags=[resolution.value, component]
            )
        except Exception:
            # Database logging failed, continue without it
            pass

    # Convenience methods for different resolutions
    def level1(self, level: LogLevel, component: str, message: str, **context):
        """Log level 1 (top-level events)"""
        self.log(level, LogResolution.LEVEL1, component, message, **context)

    def level2(self, level: LogLevel, component: str, message: str, **context):
        """Log level 2 (component interactions)"""
        self.log(level, LogResolution.LEVEL2, component, message, **context)

    def level3(self, level: LogLevel, component: str, message: str, **context):
        """Log level 3 (detailed operations)"""
        self.log(level, LogResolution.LEVEL3, component, message, **context)

    def level4(self, level: LogLevel, component: str, message: str, **context):
        """Log level 4 (full debug)"""
        self.log(level, LogResolution.LEVEL4, component, message, **context)

    # Specific event type methods
    def thought(self, thought_text: str, component: str = "thought_layer", **context):
        """Log a thought event (Level 1)"""
        self.level1(LogLevel.INFO, component,
                   f"Thought: {thought_text[:100]}{'...' if len(thought_text) > 100 else ''}",
                   thought_full=thought_text, **context)

    def action(self, action_type: str, action_content: str, component: str = "pfc", **context):
        """Log an action event (Level 1)"""
        self.level1(LogLevel.INFO, component,
                   f"Action: {action_type} - {action_content[:100]}{'...' if len(action_content) > 100 else ''}",
                   action_type=action_type, action_content=action_content, **context)

    def world_event(self, event_description: str, component: str = "world", **context):
        """Log a world event (Level 1)"""
        self.level1(LogLevel.INFO, component,
                   f"World Event: {event_description[:100]}{'...' if len(event_description) > 100 else ''}",
                   event_description=event_description, **context)

    def memory_operation(self, operation: str, memory_type: str, component: str = "memory", **context):
        """Log a memory operation (Level 2)"""
        self.level2(LogLevel.INFO, component,
                   f"Memory {operation}: {memory_type}",
                   operation=operation, memory_type=memory_type, **context)

    def api_call(self, api_name: str, endpoint: str, component: str = "api", **context):
        """Log an API call (Level 3)"""
        self.level3(LogLevel.DEBUG, component,
                   f"API Call: {api_name} -> {endpoint}",
                   api_name=api_name, endpoint=endpoint, **context)

    def neurochemistry_change(self, changes: Dict[str, float], component: str = "neurochemistry", **context):
        """Log neurochemistry changes (Level 2)"""
        self.level2(LogLevel.DEBUG, component,
                   f"Neurochemistry: {', '.join([f'{k}:{v:.2f}' for k,v in changes.items()])}",
                   neurochemistry_changes=changes, **context)

    def error(self, error_message: str, component: str, exception: Exception = None, **context):
        """Log an error (all levels)"""
        context.update({'exception': str(exception) if exception else None})
        for resolution in LogResolution:
            self.log(LogLevel.ERROR, resolution, component, error_message, **context)

    def performance_metric(self, metric_name: str, value: float, unit: str = "",
                          component: str = "performance", **context):
        """Log a performance metric (Level 2)"""
        self.level2(LogLevel.INFO, component,
                   f"Performance: {metric_name} = {value} {unit}",
                   metric_name=metric_name, metric_value=value, unit=unit, **context)

        # Also save to performance metrics table
        try:
            from .persistent_memory import get_memory_manager
            memory_manager = get_memory_manager()
            with memory_manager.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO performance_metrics
                        (component, metric_name, metric_value, unit, context)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (component, metric_name, value, unit, json.dumps(context)))
                    conn.commit()
        except Exception:
            pass

    def autobiographical_narrative(self, narrative_content: str, change_type: str = "update", 
                                 component: str = "autobiographical_memory", **context):
        """Log autobiographical narrative changes to dedicated file"""
        timestamp = datetime.now().isoformat()
        
        # Create dedicated autobiographical log file
        narrative_dir = self.log_dir / "specialized"
        narrative_dir.mkdir(exist_ok=True)
        narrative_file = narrative_dir / f"autobiographical_narrative_{self.session_id}.log"
        
        narrative_entry = {
            'timestamp': timestamp,
            'change_type': change_type,
            'narrative_content': narrative_content,
            'component': component,
            'session_id': self.session_id,
            **context
        }
        
        # Write to dedicated file
        with open(narrative_file, 'a', encoding='utf-8') as f:
            f.write(f"{json.dumps(narrative_entry, indent=2)}\n---\n")
            f.flush()
        
        # Also log at level 1 for main logging
        self.level1(LogLevel.INFO, component,
                   f"Autobiographical {change_type}: {narrative_content[:150]}{'...' if len(narrative_content) > 150 else ''}",
                   narrative_content=narrative_content, change_type=change_type, **context)

    def personality_self_model(self, personality_changes: Dict[str, Any], change_type: str = "update",
                              component: str = "self_model", **context):
        """Log personality self-model changes to dedicated file"""
        timestamp = datetime.now().isoformat()
        
        # Create dedicated personality log file
        personality_dir = self.log_dir / "specialized"
        personality_dir.mkdir(exist_ok=True)
        personality_file = personality_dir / f"personality_self_model_{self.session_id}.log"
        
        personality_entry = {
            'timestamp': timestamp,
            'change_type': change_type,
            'personality_changes': personality_changes,
            'component': component,
            'session_id': self.session_id,
            **context
        }
        
        # Write to dedicated file
        with open(personality_file, 'a', encoding='utf-8') as f:
            f.write(f"{json.dumps(personality_entry, indent=2)}\n---\n")
            f.flush()
        
        # Also log at level 1 for main logging
        changes_summary = ', '.join([f"{k}: {v}" for k, v in personality_changes.items() if isinstance(v, (str, int, float))])
        self.level1(LogLevel.INFO, component,
                   f"Personality {change_type}: {changes_summary[:150]}{'...' if len(changes_summary) > 150 else ''}",
                   personality_changes=personality_changes, change_type=change_type, **context)

    def neurochemistry_state(self, nt_levels: Dict[str, float], change_trigger: str = "", 
                           component: str = "neurochemistry", **context):
        """Enhanced neurochemistry logging with state changes"""
        timestamp = datetime.now().isoformat()
        
        # Create dedicated neurochemistry log file
        neuro_dir = self.log_dir / "specialized"
        neuro_dir.mkdir(exist_ok=True)
        neuro_file = neuro_dir / f"neurochemistry_states_{self.session_id}.log"
        
        neuro_entry = {
            'timestamp': timestamp,
            'nt_levels': nt_levels,
            'change_trigger': change_trigger,
            'component': component,
            'session_id': self.session_id,
            **context
        }
        
        # Write to dedicated file
        with open(neuro_file, 'a', encoding='utf-8') as f:
            f.write(f"{json.dumps(neuro_entry, indent=2)}\n---\n")
            f.flush()
        
        # Enhanced main logging with NT levels
        nt_summary = ', '.join([f"{k}={v:.2f}" for k, v in nt_levels.items()])
        self.level1(LogLevel.INFO, component, f"NT_LEVELS: {nt_summary}")
        if change_trigger:
            self.level3(LogLevel.DEBUG, component, f"NT_TRIGGER: {change_trigger}", **context)

# Global logger instance
_logger = None

def get_logger() -> ACILogger:
    """Get global logger instance"""
    global _logger
    if _logger is None:
        _logger = ACILogger()
    return _logger

def setup_logging_for_component(component_name: str) -> ACILogger:
    """Setup logging for a specific component"""
    logger = get_logger()
    # Create component-specific logger
    component_logger = logging.getLogger(f"aci.component.{component_name}")
    return logger
