"""Configuration, prompt sets, and constants for benchmarking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model family detection — controls which API params are safe to send
# ---------------------------------------------------------------------------

def _is_o_series(model: str) -> bool:
    """Return True for reasoning models: o1, o1-mini, o1-preview, o3, o3-mini, o3-pro, o4-mini, etc."""
    m = model.lower()
    return bool(re.match(r"^o[0-9]", m))


def _is_gpt5(model: str) -> bool:
    """Return True for GPT-5 family models."""
    return model.lower().startswith("gpt-5")


@dataclass
class ModelCapabilities:
    """What a model family supports — drives parameter selection."""
    supports_temperature: bool = True
    supports_streaming: bool = True
    system_role: str = "system"       # "system" or "developer"
    max_tokens_key: str = "max_completion_tokens"  # param name for token limit

    @staticmethod
    def for_model(model: str) -> "ModelCapabilities":
        if _is_o_series(model):
            return ModelCapabilities(
                supports_temperature=False,
                supports_streaming=True,   # o3/o4 series support streaming; o1 may not but API returns clear error
                system_role="developer",
                max_tokens_key="max_completion_tokens",
            )
        if _is_gpt5(model):
            return ModelCapabilities(
                supports_temperature=True,
                supports_streaming=True,
                system_role="developer",
                max_tokens_key="max_completion_tokens",
            )
        # GPT-4.1, GPT-4o, GPT-4, etc.
        return ModelCapabilities(
            supports_temperature=True,
            supports_streaming=True,
            system_role="system",
            max_tokens_key="max_completion_tokens",
        )

# ---------------------------------------------------------------------------
# Prompt catalogue — diverse lengths & domains to stress-test fairly
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS: dict[str, dict] = {
    "short": {
        "system": "You are a concise assistant.",
        "user": "What is 2+2?",
        "label": "Short (trivial)",
    },
    "medium": {
        "system": "You are a helpful assistant.",
        "user": (
            "Explain the difference between TCP and UDP in networking. "
            "Cover reliability, ordering, use-cases, and performance trade-offs."
        ),
        "label": "Medium (technical)",
    },
    "long": {
        "system": "You are an expert technical writer.",
        "user": (
            "Write a detailed tutorial on building a REST API with Python and FastAPI. "
            "Cover project setup, routing, request validation with Pydantic, dependency injection, "
            "authentication with OAuth2, database integration with SQLAlchemy, error handling, "
            "testing with pytest, and deployment considerations. Include code examples for each section."
        ),
        "label": "Long (generation-heavy)",
    },
    "code": {
        "system": "You are an expert Python developer.",
        "user": "Write a Python function that implements a thread-safe LRU cache with TTL expiry.",
        "label": "Code generation",
    },
    "reasoning": {
        "system": "You are a logical reasoning expert.",
        "user": (
            "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river "
            "in a boat that can only carry him and one item at a time. If left alone, the fox "
            "will eat the chicken and the chicken will eat the grain. How does the farmer get "
            "everything across safely? Explain step by step."
        ),
        "label": "Reasoning / multi-step",
    },
}

# Prompt used for cache warm/cold testing (must be identical across runs)
# Azure prompt caching requires ≥1,024 input tokens with the first 1,024
# being identical.  We pad with a long system context to exceed that threshold.
CACHE_TEST_PROMPT = {
    "system": (
        "You are a highly knowledgeable assistant specializing in astronomy, astrophysics, "
        "planetary science, and space exploration. You have deep expertise in the formation "
        "and evolution of planetary systems, stellar nucleosynthesis, cosmic microwave background "
        "radiation, dark matter and dark energy, gravitational wave detection, exoplanet "
        "characterization, space mission design, orbital mechanics, and the history of human "
        "spaceflight.\n\n"
        "When answering questions, you should provide accurate, detailed, and well-structured "
        "responses. You should cite relevant scientific principles, reference notable missions "
        "and discoveries, and explain complex concepts in an accessible way. You should cover "
        "both historical context and current scientific understanding.\n\n"
        "CONTEXT AND REFERENCE MATERIAL:\n\n"
        "The Solar System formed approximately 4.6 billion years ago from the gravitational "
        "collapse of a giant interstellar molecular cloud. The vast majority of the system's "
        "mass is in the Sun, with the majority of the remaining mass contained in Jupiter. "
        "The four smaller inner system planets — Mercury, Venus, Earth, and Mars — are "
        "terrestrial planets, being primarily composed of rock and metal. The four outer system "
        "planets are giant planets, being substantially more massive than the terrestrials. "
        "The two largest outer planets, Jupiter and Saturn, are gas giants, being composed "
        "mainly of hydrogen and helium; the two outermost planets, Uranus and Neptune, are "
        "ice giants, being composed mostly of substances with relatively high melting points "
        "compared with hydrogen and helium, called volatiles, such as water, ammonia, and "
        "methane. All eight planets have nearly circular orbits that lie near the plane of "
        "Earth's orbit, called the ecliptic.\n\n"
        "Mercury is the smallest and closest planet to the Sun. Its orbital period around the "
        "Sun of 87.97 Earth days is the shortest of all the planets. Mercury is one of four "
        "terrestrial planets in the Solar System, and is a rocky body like Earth. It has a "
        "diameter of 4,879.4 km at its equator, which is about 38% of Earth's diameter. "
        "Mercury has no natural satellites. The planet has a significant, and apparently "
        "globally distributed, parsing of the surface indicates volcanic plains that cover "
        "approximately 60% of the surface observed. The surface also reveals extensive "
        "compressive tectonic features consistent with global contraction.\n\n"
        "Venus is the second planet from the Sun. It is sometimes called Earth's sister planet "
        "because of their similar size, mass, proximity to the Sun, and bulk composition. It "
        "is radically different from Earth in other respects. It has the densest atmosphere of "
        "the four terrestrial planets, consisting of more than 96% carbon dioxide. The "
        "atmospheric pressure at the planet's surface is about 92 times the sea level pressure "
        "of Earth. Venus has an extremely hot surface with a mean temperature of 737 K (464 °C; "
        "867 °F), making it the hottest planet in the Solar System. Venus has no natural "
        "satellites. It is the second-brightest natural object in Earth's night sky after the "
        "Moon, bright enough to cast shadows at night and visible to the naked eye in daylight.\n\n"
        "Earth is the third planet from the Sun and the only astronomical object known to harbor "
        "life. While large volumes of water can be found throughout the Solar System, only Earth "
        "sustains liquid surface water. About 71% of Earth's surface is made up of the ocean, "
        "dwarfing Earth's polar ice, lakes, and rivers. The remaining 29% of Earth's surface is "
        "land, consisting of continents and islands. Earth's surface layer is formed of several "
        "slowly moving tectonic plates, interacting to produce mountain ranges, volcanoes, and "
        "earthquakes. Earth's liquid outer core generates the magnetic field that shapes Earth's "
        "magnetosphere, deflecting destructive solar winds.\n\n"
        "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar "
        "System, being larger than only Mercury. Mars carries the name of the Roman god of war "
        "and is often called the Red Planet. Mars is a terrestrial planet with a thin atmosphere "
        "(less than 1% that of Earth's), and has a crust primarily composed of elements similar "
        "to Earth's crust, as well as a core made of iron and nickel. Mars has surface features "
        "such as impact craters, valleys, dunes, and polar ice caps. Mars has two small, "
        "irregularly shaped moons: Phobos and Deimos.\n\n"
        "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a "
        "gas giant with a mass more than two and a half times that of all the other planets in "
        "the Solar System combined, and slightly less than one one-thousandth the mass of the "
        "Sun. Jupiter is the third brightest natural object in the Earth's night sky after the "
        "Moon and Venus, and it has been observed since prehistoric times. It was named after "
        "Jupiter, the chief deity of ancient Roman religion. Jupiter has 95 known moons and "
        "most likely has many more, including the four large Galilean moons discovered by "
        "Galileo Galilei in 1610.\n\n"
        "Saturn is the sixth planet from the Sun and the second-largest in the Solar System, "
        "after Jupiter. It is a gas giant with an average radius of about nine and a half times "
        "that of Earth. It has only one-eighth the average density of Earth; however, with its "
        "larger volume, Saturn is over 95 times more massive. Saturn's interior is most likely "
        "composed of a rocky core, surrounded by a deep layer of metallic hydrogen, an "
        "intermediate layer of liquid hydrogen and liquid helium, and finally a gaseous outer "
        "layer. Saturn has a prominent ring system that consists of nine continuous main rings "
        "and three discontinuous arcs, composed mostly of ice particles, with a smaller amount "
        "of rocky debris and dust.\n\n"
        "Uranus is the seventh planet from the Sun. Its name is a reference to the Greek god of "
        "the sky, Uranus, who, according to Greek mythology, was the great-grandfather of Ares "
        "(Mars), grandfather of Zeus (Jupiter), and father of Cronus (Saturn). Uranus has the "
        "third-largest planetary radius and fourth-largest planetary mass in the Solar System. "
        "The planet is similar in composition to Neptune, and both have bulk chemical compositions "
        "which differ from that of the larger gas giants Jupiter and Saturn. For this reason, "
        "scientists often classify Uranus and Neptune as ice giants to distinguish them from "
        "the other giant planets.\n\n"
        "Neptune is the eighth planet from the Sun and the farthest known solar planet. In the "
        "Solar System, it is the fourth-largest planet by diameter, the third-most-massive planet, "
        "and the densest giant planet. It is 17 times the mass of Earth, and slightly more massive "
        "than its near-twin Uranus. Neptune is denser and physically smaller than Uranus because "
        "its greater mass causes more gravitational compression of its atmosphere. It is referred "
        "to as one of the solar system's two ice giant planets along with Uranus.\n\n"
        "END OF REFERENCE MATERIAL.\n\n"
        "Using the above reference material better inform and ground your answer."
    ),
    "user": "List the planets in our solar system in order from the sun, with one interesting fact about each.",
}


@dataclass
class BenchmarkConfig:
    """Runtime configuration for a benchmark session."""

    model: str = "gpt-4.1-nano"
    iterations: int = 3
    warmup: int = 1
    max_tokens: int = 512
    temperature: float = 0.7
    concurrency: int = 5
    prompt_keys: list[str] = field(default_factory=lambda: list(BENCHMARK_PROMPTS.keys()))
    stream: bool = True
    cache_rounds: int = 5
    timeout: float = 120.0
