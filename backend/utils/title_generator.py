"""
Title generator for conversations - creates concise 2-3 word titles from prompts.
Similar to how ChatGPT generates conversation titles.
"""

import re
from typing import Optional, List, Tuple


def generate_title(prompt: str, max_words: int = 3) -> str:
    """
    Generate a concise title (2-3 words) from a user prompt.
    
    Args:
        prompt: The user's prompt/message
        max_words: Maximum number of words in the title (default: 3)
    
    Returns:
        A clean, title-cased string of 2-3 words
    
    Examples:
        "What is the current stock price of Apple?" -> "Apple Stock Price"
        "Analyze this document for me" -> "Document Analysis"
        "get me news on trump" -> "Trump News"
    """
    
    # Remove URLs and email addresses
    cleaned = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', prompt)
    
    # Remove special characters except spaces and hyphens
    cleaned = re.sub(r'[^\w\s\-]', '', cleaned)
    
    # Split into words and remove empty strings
    words = [w.strip() for w in cleaned.split() if w.strip()]
    
    if not words:
        return "Untitled"
    
    # Remove common stop words (articles, prepositions, etc.)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'from', 'with', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'me', 'my', 'you', 'this',
        'that', 'what', 'which', 'who', 'where', 'when', 'why', 'how', 'get',
        'give', 'make', 'take', 'please', 'pls', 'thx', 'thanks',
        # add pronouns and generic verbs that cause vague titles
        'i', 'we', 'us', 'our', 'it', 'they', 'them', 'their',
        'analyze', 'analyse', 'analysis', 'execute', 'execution', 'requested', 'request', 'start'
    }
    
    # Filter out stop words
    filtered_words = [w for w in words if w.lower() not in stop_words]
    
    # If all words were filtered, use first non-empty words
    if not filtered_words:
        filtered_words = words[:max_words]
    
    # Take first max_words
    title_words = filtered_words[:max_words]
    
    # Join and title-case
    title = ' '.join(title_words).title()
    
    # Ensure we have at least one word
    if not title:
        title = "Untitled"
    
    return title


def _extract_named_entities(prompt: str) -> List[str]:
    """Very lightweight proper-noun extractor: returns capitalized word sequences."""
    # Find sequences of Capitalized words (e.g., "Jensen Huang", "New York", "Tesla")
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", prompt)
    # Filter generic words
    blacklist = {"I", "The", "A", "An"}
    return [c for c in candidates if c not in blacklist][:3]


def generate_improved_title(prompt: str, tasks: list = None, max_words: int = 3) -> str:
    """
    Generate a more intelligent title by also considering parsed tasks.
    
    Args:
        prompt: The user's original prompt
        tasks: List of parsed tasks (if available)
        max_words: Maximum number of words (default: 3)
    
    Returns:
        A clean title string
    
    Examples:
        If tasks include ["get_stock_price"], title might be "Stock Price Check"
        If tasks include ["analyze_document"], title might be "Document Analysis"
    """
    
    # If we have tasks, try to use them for better titles
    if tasks and len(tasks) > 0:
        task_keywords = {}
        
        # Map task types to keywords
        task_to_title = {
            'get_stock_price': 'Stock Quote',
            'get_news': 'News',
            'weather': 'Weather Check',
            'document_analysis': 'Doc Analysis',
            'image_analysis': 'Image Analysis',
            'search': 'Search Query',
            'translate': 'Translation',
            'summarize': 'Summarization',
        }
        
        # Extract first task and use mapped title if available
        first_task = tasks[0].get('task_name', '') if isinstance(tasks[0], dict) else str(tasks[0])
        
        for task_key, title_value in task_to_title.items():
            if task_key.lower() in first_task.lower():
                # Specialize "News" titles with an entity if present
                if title_value == 'News':
                    entities = _extract_named_entities(prompt)
                    if entities:
                        return f"{entities[0]} News"
                    # Fallback to generic News Update with a key term from prompt
                    keywords = [w for w in re.findall(r"[A-Za-z]+", prompt) if len(w) > 2]
                    if keywords:
                        return f"{keywords[0].title()} News"
                    return "Latest News"
                return title_value
    
    # Fall back to prompt-based title
    # If prompt includes explicit news wording, try to extract entity for better titles
    if 'news' in prompt.lower():
        entities = _extract_named_entities(prompt)
        if entities:
            return f"{entities[0]} News"
        return "Latest News"

    return generate_title(prompt, max_words)


# Test the function
if __name__ == "__main__":
    test_prompts = [
        "What is the current stock price of Apple?",
        "Analyze this document for me",
        "get me news on trump",
        "Can you translate this to French?",
        "What's the weather like in New York?",
        "Summarize this article for me",
        "Create a plan for the project",
    ]
    
    print("Title Generation Examples:")
    print("-" * 50)
    for prompt in test_prompts:
        title = generate_title(prompt)
        print(f"Prompt: {prompt}")
        print(f"Title:  {title}")
        print()
