"""
Agents Router - Handles agent marketplace endpoints.

Extracted from main.py to improve code organization and maintainability.
Includes: registration, search, get, rate.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status, Query, Response, Body
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, select

from database import SessionLocal
from models import Agent, AgentCapability, AgentEndpoint, EndpointParameter
from backend.schemas import AgentCard

router = APIRouter(prefix="/api/agents", tags=["Agents"])
logger = logging.getLogger("uvicorn.error")

# --- Database Dependency (duplicated for now, will be centralized later) ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Sentence Transformer Model Loading (duplicated for now) ---
_model = None

def get_sentence_transformer_model():
    """Lazy load the sentence transformer model only when needed"""
    import os
    global _model
    if _model is None:
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['JAX_PLATFORMS'] = ''
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-mpnet-base-v2')
    return _model


@router.post("/register", response_model=AgentCard)
def register_or_update_agent(agent_data: AgentCard, response: Response, db: Session = Depends(get_db)):
    db_agent = db.query(Agent).options(
        joinedload(Agent.capability_vectors),
        joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
    ).get(agent_data.id)

    agent_dict = agent_data.model_dump(
        mode='json',
        exclude={"endpoints"},
        exclude_none=True,
        exclude_unset=True
    )

    if db_agent:
        for key, value in agent_dict.items():
            setattr(db_agent, key, value)
        db_agent.capability_vectors.clear()
        db_agent.endpoints.clear()
        response.status_code = status.HTTP_200_OK
    else:
        db_agent = Agent(**agent_dict)
        db.add(db_agent)
        response.status_code = status.HTTP_201_CREATED

    if agent_data.capabilities:
        sentence_model = get_sentence_transformer_model()
        for cap_text in agent_data.capabilities:
            embedding_vector = sentence_model.encode(cap_text)
            new_capability = AgentCapability(
                agent=db_agent,
                capability_text=cap_text,
                embedding=embedding_vector
            )
            db.add(new_capability)

    if agent_data.endpoints:
        for endpoint_data in agent_data.endpoints:
            new_endpoint = AgentEndpoint(
                agent=db_agent,
                endpoint=str(endpoint_data.endpoint),
                http_method=endpoint_data.http_method,
                description=endpoint_data.description
            )
            db.add(new_endpoint)

            if endpoint_data.parameters:
                for param_data in endpoint_data.parameters:
                    new_param = EndpointParameter(
                        endpoint=new_endpoint,
                        name=param_data.name,
                        description=param_data.description,
                        param_type=param_data.param_type,
                        required=param_data.required,
                        default_value=param_data.default_value
                    )
                    db.add(new_param)

    db.commit()
    db.refresh(db_agent)
    return AgentCard.model_validate(db_agent)


@router.get("/search", response_model=List[AgentCard])
def search_agents(
    db: Session = Depends(get_db),
    capabilities: List[str] = Query(..., description="A list of task names to find capable agents for."),
    max_price: Optional[float] = Query(None),
    min_rating: Optional[float] = Query(None),
    similarity_threshold: float = Query(0.5, description="Cosine distance threshold (lower is stricter).")
):
    """
    Finds active agents that match ANY of the specified capabilities using vector search.
    Falls back to text search if vector search fails.
    """
    if not capabilities:
        return []

    try:
        sentence_model = get_sentence_transformer_model()
        conditions = []
        for task_name in capabilities:
            query_vector = sentence_model.encode(task_name)
            subquery = select(AgentCapability.agent_id).where(
                AgentCapability.embedding.cosine_distance(query_vector) < similarity_threshold
            )
            conditions.append(Agent.id.in_(subquery))

        query = db.query(Agent).options(
            joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
        ).filter(Agent.status == 'active').filter(or_(*conditions))

        if max_price is not None:
            query = query.filter(Agent.price_per_call_usd <= max_price)
        if min_rating is not None:
            query = query.filter(Agent.rating >= min_rating)

        return query.all()
    
    except Exception as e:
        logger.warning(f"Vector search failed, falling back to text search: {e}")
        query = db.query(Agent).options(
            joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
        ).filter(Agent.status == 'active')
        
        if max_price is not None:
            query = query.filter(Agent.price_per_call_usd <= max_price)
        if min_rating is not None:
            query = query.filter(Agent.rating >= min_rating)
        
        return query.all()


@router.get("/all", response_model=List[AgentCard])
def get_all_agents(db: Session = Depends(get_db)):
    """Returns all agents in the agents table as a JSON list."""
    return db.query(Agent).options(
        joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
    ).all()


@router.get("/{agent_id}", response_model=AgentCard)
def get_agent(agent_id: str, db: Session = Depends(get_db)):
    db_agent = db.query(Agent).options(
        joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
    ).get(agent_id)
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found!")
    return db_agent


@router.post("/{agent_id}/rate", response_model=AgentCard)
def rate_agent(agent_id: str, rating: float = Body(..., embed=True), db: Session = Depends(get_db)):
    """Update the agent's rating as the mean of the current rating and the new user rating."""
    db_agent = db.get(Agent, agent_id)
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found!")
    if rating < 0 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5.")
    
    current_rating = db_agent.rating if db_agent.rating is not None else 0.0
    count = db_agent.rating_count if db_agent.rating_count is not None else 0
    new_rating = ((current_rating * count) + rating) / (count + 1) if count > 0 else rating
    db_agent.rating = float(new_rating)
    db_agent.rating_count = int(count + 1)
    db.commit()
    db.refresh(db_agent)
    return AgentCard.model_validate(db_agent)


@router.post("/by-name/{agent_name}/rate", response_model=AgentCard)
def rate_agent_by_name(agent_name: str, rating: float = Body(..., embed=True), db: Session = Depends(get_db)):
    """Update the agent's rating using the agent's name as a fallback."""
    db_agent = db.query(Agent).filter(Agent.name == agent_name).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found!")
    if rating < 0 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5.")
    
    current_rating = db_agent.rating if db_agent.rating is not None else 0.0
    count = db_agent.rating_count if db_agent.rating_count is not None else 0
    new_rating = ((current_rating * count) + rating) / (count + 1) if count > 0 else rating
    db_agent.rating = float(new_rating)
    db_agent.rating_count = int(count + 1)
    db.commit()
    db.refresh(db_agent)
    return AgentCard.model_validate(db_agent)
