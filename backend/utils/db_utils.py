"""
Database Session Utilities

Provides standardized database session management utilities.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@contextmanager
def get_db_session(session_factory) -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Ensures proper session lifecycle management:
    - Automatically commits on success
    - Automatically rolls back on exception
    - Always closes the session

    Args:
        session_factory: Callable that returns a new Session (e.g., SessionLocal)

    Yields:
        SQLAlchemy Session object

    Example:
        with get_db_session(SessionLocal) as db:
            result = db.query(MyModel).all()
    """
    db = session_factory()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()


@contextmanager
def get_db_session_optional(
    existing_session: Optional[Session], session_factory
) -> Generator[Session, None, None]:
    """
    Context manager that optionally uses an existing session or creates a new one.

    This is useful for functions that may be called with or without an existing
    database session, avoiding the `should_close_db` pattern.

    Args:
        existing_session: An existing session to use, or None to create a new one
        session_factory: Callable that returns a new Session

    Yields:
        SQLAlchemy Session object (either existing or new)

    Example:
        def my_function(db: Session = None):
            with get_db_session_optional(db, SessionLocal) as session:
                return session.query(MyModel).all()
    """
    if existing_session is not None:
        # Use existing session, don't close it
        yield existing_session
    else:
        # Create new session and manage it
        with get_db_session(session_factory) as db:
            yield db


def safe_query_first(session: Session, query, error_msg: str = "Query failed"):
    """
    Execute a query safely, returning None on failure instead of raising.

    Args:
        session: SQLAlchemy session
        query: Query to execute
        error_msg: Error message to log on failure

    Returns:
        First result or None
    """
    try:
        return query.first()
    except Exception as e:
        logger.error(f"{error_msg}: {e}")
        return None


def safe_query_all(session: Session, query, error_msg: str = "Query failed"):
    """
    Execute a query safely, returning empty list on failure instead of raising.

    Args:
        session: SQLAlchemy session
        query: Query to execute
        error_msg: Error message to log on failure

    Returns:
        List of results or empty list
    """
    try:
        return query.all()
    except Exception as e:
        logger.error(f"{error_msg}: {e}")
        return []
