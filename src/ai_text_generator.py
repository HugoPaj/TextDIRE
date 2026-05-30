"""
AI Text Generator for Text-DIRE experiments.

Generates balanced datasets from multiple AI models using OpenAI and Anthropic APIs.
Supports GPT-5.2, GPT-5-mini, Claude Sonnet 4.5, and Claude Haiku 4.5.
"""

import os
import json
import time
import hashlib
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime


def load_project_env(env_path: Optional[Path] = None, verbose: bool = False) -> bool:
    """Load the project .env file when explicitly requested."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        return False

    loaded = load_dotenv(env_path)
    if verbose and loaded:
        print(f"Loaded environment from {env_path}")
    return loaded


@dataclass
class GenerationConfig:
    """Configuration for AI text generation."""
    model: str
    prompt_type: str
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0


@dataclass
class GeneratedText:
    """Container for a generated text sample."""
    text: str
    model: str
    prompt_type: str
    prompt: str
    temperature: float
    timestamp: str
    token_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Model configurations
OPENAI_MODELS = {
    "gpt-5.2": "gpt-5.2",
    "gpt-5-mini": "gpt-5-mini",
    # Fallback models if GPT-5 not available
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}

ANTHROPIC_MODELS = {
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "claude-haiku-4.5": "claude-haiku-4-5-20251009",
    # Fallback models
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-haiku-3.5": "claude-3-5-haiku-20241022",
}

# Prompt templates for different generation scenarios
PROMPT_TEMPLATES = {
    "qa": [
        "What is {topic}? Explain in detail.",
        "How does {topic} work? Provide a comprehensive explanation.",
        "Can you explain the concept of {topic}?",
        "What are the main aspects of {topic}?",
        "Describe {topic} and its significance.",
    ],
    "essay": [
        "Write a short essay about {topic}.",
        "Discuss the importance of {topic} in modern society.",
        "Analyze the impact of {topic} on our daily lives.",
        "Explore the relationship between {topic} and technology.",
        "Examine the future of {topic}.",
    ],
    "creative": [
        "Write a short story involving {topic}.",
        "Create a narrative scene featuring {topic}.",
        "Compose a creative piece inspired by {topic}.",
        "Write a descriptive passage about {topic}.",
        "Craft an imaginative tale about {topic}.",
    ],
    "technical": [
        "Explain the technical aspects of {topic}.",
        "Describe the implementation details of {topic}.",
        "What are the engineering challenges in {topic}?",
        "How is {topic} designed and built?",
        "Discuss the architecture of {topic}.",
    ],
    "code_doc": [
        "Write documentation for a function that implements {topic}.",
        "Document a class that handles {topic}.",
        "Create API documentation for a module dealing with {topic}.",
        "Write a technical specification for {topic}.",
        "Document the design decisions for implementing {topic}.",
    ],
}

# Diverse topics for generation
TOPICS = [
    # Science & Technology
    "artificial intelligence", "machine learning", "quantum computing",
    "blockchain technology", "renewable energy", "climate change",
    "space exploration", "genetic engineering", "nanotechnology",
    "cybersecurity", "virtual reality", "autonomous vehicles",

    # Society & Culture
    "globalization", "urbanization", "digital privacy",
    "social media influence", "remote work", "education reform",
    "healthcare systems", "economic inequality", "cultural diversity",
    "democracy", "human rights", "sustainable development",

    # History & Philosophy
    "ancient civilizations", "the Renaissance", "industrial revolution",
    "world wars", "philosophy of mind", "ethics in technology",
    "existentialism", "scientific method", "history of mathematics",

    # Nature & Environment
    "biodiversity", "ocean ecosystems", "forest conservation",
    "wildlife protection", "pollution control", "sustainable agriculture",
    "weather patterns", "geological formations", "evolutionary biology",

    # Arts & Humanities
    "modern art", "classical music", "literature evolution",
    "film history", "architecture styles", "photography techniques",
    "theater traditions", "dance forms", "culinary arts",
]


class AITextGenerator:
    """
    Multi-model AI text generator using OpenAI and Anthropic APIs.

    Usage:
        generator = AITextGenerator()
        texts = generator.generate_dataset(
            models=["gpt-5.2", "claude-sonnet-4.5"],
            num_samples=500,
            prompt_types=["qa", "essay", "creative"]
        )
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        rate_limit_delay: float = 0.5,
        load_env: bool = True,
    ):
        """
        Initialize the AI text generator.

        Args:
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            anthropic_api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            cache_dir: Directory for caching generated texts
            rate_limit_delay: Delay between API calls in seconds
            load_env: Whether to load the project .env file before reading env vars
        """
        if load_env:
            load_project_env()

        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.rate_limit_delay = rate_limit_delay

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "ai_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clients lazily
        self._openai_client = None
        self._anthropic_client = None

    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._openai_client

    @property
    def anthropic_client(self):
        """Lazy initialization of Anthropic client."""
        if self._anthropic_client is None:
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
            try:
                from anthropic import Anthropic
                self._anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._anthropic_client

    def _get_cache_key(self, model: str, prompt: str, temperature: float) -> str:
        """Generate a cache key for a specific generation request."""
        content = f"{model}:{prompt}:{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[GeneratedText]:
        """Load a cached generation if it exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return GeneratedText(**data)
            except Exception:
                return None
        return None

    def _save_to_cache(self, cache_key: str, generated: GeneratedText):
        """Save a generation to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(generated.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to cache generation: {e}")

    def _generate_openai(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Optional[GeneratedText]:
        """Generate text using OpenAI API."""
        model_id = OPENAI_MODELS.get(model, model)

        try:
            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide detailed, well-written responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = response.choices[0].message.content
            token_count = response.usage.completion_tokens if response.usage else None

            return GeneratedText(
                text=text,
                model=model,
                prompt_type="",  # Will be set by caller
                prompt=prompt,
                temperature=temperature,
                timestamp=datetime.now().isoformat(),
                token_count=token_count,
                metadata={"model_id": model_id, "provider": "openai"}
            )

        except Exception as e:
            print(f"OpenAI generation error ({model}): {e}")
            return None

    def _generate_anthropic(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Optional[GeneratedText]:
        """Generate text using Anthropic API."""
        model_id = ANTHROPIC_MODELS.get(model, model)

        try:
            response = self.anthropic_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a helpful assistant. Provide detailed, well-written responses."
            )

            text = response.content[0].text
            token_count = response.usage.output_tokens if response.usage else None

            return GeneratedText(
                text=text,
                model=model,
                prompt_type="",  # Will be set by caller
                prompt=prompt,
                temperature=temperature,
                timestamp=datetime.now().isoformat(),
                token_count=token_count,
                metadata={"model_id": model_id, "provider": "anthropic"}
            )

        except Exception as e:
            print(f"Anthropic generation error ({model}): {e}")
            return None

    def generate(
        self,
        model: str,
        prompt: str,
        prompt_type: str = "qa",
        temperature: float = 0.7,
        max_tokens: int = 512,
        use_cache: bool = True,
    ) -> Optional[GeneratedText]:
        """
        Generate a single text sample.

        Args:
            model: Model name (e.g., "gpt-5.2", "claude-sonnet-4.5")
            prompt: The generation prompt
            prompt_type: Type of prompt (qa, essay, creative, technical, code_doc)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching

        Returns:
            GeneratedText object or None if generation failed
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(model, prompt, temperature)
            cached = self._load_from_cache(cache_key)
            if cached:
                cached.prompt_type = prompt_type
                return cached

        # Determine which API to use
        if model in OPENAI_MODELS or model.startswith("gpt"):
            generated = self._generate_openai(model, prompt, temperature, max_tokens)
        elif model in ANTHROPIC_MODELS or model.startswith("claude"):
            generated = self._generate_anthropic(model, prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown model: {model}")

        if generated:
            generated.prompt_type = prompt_type

            # Save to cache
            if use_cache:
                cache_key = self._get_cache_key(model, prompt, temperature)
                self._save_to_cache(cache_key, generated)

            # Rate limiting
            time.sleep(self.rate_limit_delay)

        return generated

    def generate_batch(
        self,
        model: str,
        num_samples: int,
        prompt_types: Optional[list[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[GeneratedText]:
        """
        Generate a batch of text samples from a single model.

        Args:
            model: Model name
            num_samples: Number of samples to generate
            prompt_types: List of prompt types to use (defaults to all)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of GeneratedText objects
        """
        if prompt_types is None:
            prompt_types = list(PROMPT_TEMPLATES.keys())

        results = []
        samples_per_type = num_samples // len(prompt_types)
        remainder = num_samples % len(prompt_types)

        current = 0

        for type_idx, prompt_type in enumerate(prompt_types):
            # Distribute remainder samples
            type_samples = samples_per_type + (1 if type_idx < remainder else 0)
            templates = PROMPT_TEMPLATES[prompt_type]

            for i in range(type_samples):
                # Select template and topic
                template = templates[i % len(templates)]
                topic = TOPICS[i % len(TOPICS)]
                prompt = template.format(topic=topic)

                generated = self.generate(
                    model=model,
                    prompt=prompt,
                    prompt_type=prompt_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if generated:
                    results.append(generated)

                current += 1
                if progress_callback:
                    progress_callback(current, num_samples)

        return results

    def generate_dataset(
        self,
        models: Optional[list[str]] = None,
        num_samples_per_model: int = 500,
        prompt_types: Optional[list[str]] = None,
        temperatures: Optional[list[float]] = None,
        save_path: Optional[str] = None,
    ) -> dict[str, list[GeneratedText]]:
        """
        Generate a complete dataset from multiple models.

        Args:
            models: List of model names (defaults to all supported models)
            num_samples_per_model: Number of samples per model
            prompt_types: List of prompt types to use
            temperatures: List of temperatures to use (cycles through)
            save_path: Path to save the dataset JSON

        Returns:
            Dictionary mapping model names to lists of GeneratedText
        """
        if models is None:
            models = [
                "gpt-5.2", "gpt-5-mini",
                "claude-sonnet-4.5", "claude-haiku-4.5"
            ]

        if temperatures is None:
            temperatures = [0.7]

        dataset = {}

        for model in models:
            print(f"\nGenerating {num_samples_per_model} samples from {model}...")

            samples_per_temp = num_samples_per_model // len(temperatures)
            model_results = []

            for temp_idx, temp in enumerate(temperatures):
                temp_samples = samples_per_temp
                if temp_idx == len(temperatures) - 1:
                    temp_samples += num_samples_per_model % len(temperatures)

                def progress(current, total):
                    print(f"  [{model}] {current}/{total} (temp={temp})", end="\r")

                results = self.generate_batch(
                    model=model,
                    num_samples=temp_samples,
                    prompt_types=prompt_types,
                    temperature=temp,
                    progress_callback=progress,
                )
                model_results.extend(results)

            dataset[model] = model_results
            print(f"\n  Completed: {len(model_results)} samples")

        # Save dataset if path provided
        if save_path:
            self.save_dataset(dataset, save_path)

        return dataset

    def save_dataset(self, dataset: dict[str, list[GeneratedText]], path: str):
        """Save dataset to JSON file."""
        serializable = {
            model: [g.to_dict() for g in texts]
            for model, texts in dataset.items()
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        print(f"Dataset saved to {path}")

    @staticmethod
    def load_dataset(path: str) -> dict[str, list[GeneratedText]]:
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            model: [GeneratedText(**item) for item in texts]
            for model, texts in data.items()
        }


def get_available_models() -> dict[str, list[str]]:
    """Get dictionary of available models by provider."""
    return {
        "openai": list(OPENAI_MODELS.keys()),
        "anthropic": list(ANTHROPIC_MODELS.keys()),
    }


def generate_sample_prompts(
    num_prompts: int = 10,
    prompt_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Generate sample prompts for preview/testing.

    Returns:
        List of dicts with 'prompt_type', 'topic', 'prompt' keys
    """
    if prompt_types is None:
        prompt_types = list(PROMPT_TEMPLATES.keys())

    prompts = []
    for i in range(num_prompts):
        prompt_type = prompt_types[i % len(prompt_types)]
        template = PROMPT_TEMPLATES[prompt_type][i % len(PROMPT_TEMPLATES[prompt_type])]
        topic = TOPICS[i % len(TOPICS)]

        prompts.append({
            "prompt_type": prompt_type,
            "topic": topic,
            "prompt": template.format(topic=topic),
        })

    return prompts


if __name__ == "__main__":
    # Example usage
    print("AI Text Generator for Text-DIRE")
    print("=" * 50)

    print("\nAvailable models:")
    for provider, models in get_available_models().items():
        print(f"  {provider}: {', '.join(models)}")

    print("\nSample prompts:")
    for p in generate_sample_prompts(5):
        print(f"  [{p['prompt_type']}] {p['prompt'][:60]}...")

    print("\nTo generate a dataset:")
    print("  generator = AITextGenerator()")
    print("  dataset = generator.generate_dataset(num_samples_per_model=500)")
