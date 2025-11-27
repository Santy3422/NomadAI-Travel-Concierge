#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
========================================================
CAPSTONE CODEBOOK – TRAVEL CONCIERGE AGENT (SINGLE FILE)
========================================================

Track
-----
Concierge Agents

Pitch (Category 1: The Pitch)
-----------------------------
Problem:
    Planning personal travel (choosing dates, budgeting, and building
    an itinerary) is time-consuming and fragmented. You have to juggle
    preferences, budgets, destinations, and schedules manually.

Solution:
    This project implements a **Travel Concierge Agent** that:
      - Collects the user's travel goals and preferences.
      - Checks basic budget feasibility.
      - Generates a simple day-by-day itinerary.
      - Stores user preferences so future plans get faster and better.

Value:
    - Reduces manual time spent on drafting itineraries.
    - Re-uses stored preferences to speed up future trip planning.
    - Provides a simple, extensible base for more advanced agents.

Key Concepts Demonstrated (at least 3)
--------------------------------------
1. Multi-Agent System
   - PreferenceAgent: Builds/updates user travel preferences.
   - BudgetAgent    : Checks and summarizes budget feasibility.
   - ItineraryAgent : Creates a basic itinerary.
   - EvaluationAgent: Scores the final plan (simple rubric).
   The Orchestrator runs these agents sequentially.

2. Tools
   - Custom tools (pure Python; no paid APIs):
       * estimate_flight_cost
       * estimate_accommodation_cost
       * suggest_activities_for_destination
       * simple_weather_stub
   These are implemented as regular Python functions and used by agents.

3. Sessions & Memory
   - SessionManager (in-memory “session/state” per user_id).
   - LongTermMemory backed by a local JSON file (user_preferences.json)
     for long-term storage of traveler profile (budget level, style, etc.).

4. Observability (Logging + Metrics)
   - Python logging used across agents and orchestrator.
   - Per-agent execution timing and simple metrics aggregated.

(You can name these explicitly in your written submission.)

How to Use This File
--------------------
1. Save this as, for example: travel_concierge_agent.py
2. (Optional) Create a virtual env and install dependencies (only 'requests' is used; you can remove it if you don’t want HTTP at all).
3. Run from terminal:
       python travel_concierge_agent.py
4. Follow the CLI prompts to generate a simple itinerary.

Open Source / Cost Constraints
------------------------------
- No paid APIs used.
- LLM calls are stubbed via a simple LocalLLM class (rule-based).
  You can later plug in any **open source** LLM backend like:
    - Ollama (e.g., mistral, llama3)
    - Hugging Face transformers (offline models)

========================================================
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# -------------------------------------------------------------------
# LOGGING / OBSERVABILITY
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("travel_concierge")


# -------------------------------------------------------------------
# SIMPLE LLM ABSTRACTION (NO PAID APIs)
# -------------------------------------------------------------------

class LocalLLM:
    """
    A tiny rule-based / template 'LLM' placeholder.

    This lets the rest of the agent architecture look like it's
    using an LLM, but we don't depend on any paid or external APIs.

    Later, you can extend `generate` to call:
      - an Ollama server (localhost, free, open source models),
      - a local Hugging Face model,
      - or any other open-source backend.
    """

    def __init__(self, backend: str = "rule"):
        self.backend = backend

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        # Very naive "generation" for demo purposes.
        text = f"{system_prompt}\nUSER: {user_prompt}\n\n"
        if "itinerary" in user_prompt.lower():
            return (
                "Day 1: Arrival, check-in, and an easy walk around the city center.\n"
                "Day 2: Visit 2–3 key attractions in the morning, local food tour in the evening.\n"
                "Day 3: Free exploration / shopping, optional museum.\n"
                "Day 4: Day trip to a nearby attraction.\n"
                "Day 5: Relaxed morning, brunch, and departure.\n"
            )
        if "budget summary" in user_prompt.lower():
            return "The plan seems reasonably aligned with the budget for a mid-range traveler."
        if "evaluate" in user_prompt.lower():
            return "Overall, the itinerary is suitable and balanced for the given preferences."
        # Generic fallback
        return "Here is a concise response based on your input."

# -------------------------------------------------------------------
# DATA MODELS
# -------------------------------------------------------------------

@dataclass
class TravelRequest:
    user_id: str
    destination: str
    start_date: str   # ISO "YYYY-MM-DD"
    end_date: str     # ISO
    total_budget: float
    travelers: int
    style: str        # e.g., "relaxed", "packed", "family", "adventure"
    interests: List[str]


@dataclass
class TravelPreferences:
    """
    Persistent user preferences (long-term memory).
    """
    home_city: str
    typical_budget_level: str        # "low", "medium", "high"
    preferred_accommodation: str     # "hostel", "hotel", "airbnb"
    pace: str                        # "relaxed", "packed"
    top_interests: List[str]


@dataclass
class BudgetSummary:
    estimated_flight_cost: float
    estimated_accommodation_cost: float
    estimated_daily_spend: float
    total_estimated_cost: float
    is_within_budget: bool
    notes: str


@dataclass
class ItineraryResult:
    raw_text: str
    daily_plan: Dict[int, str]  # day_index -> text


@dataclass
class EvaluationResult:
    score_overall: float
    score_budget_fit: float
    score_interest_alignment: float
    comments: str


# -------------------------------------------------------------------
# MEMORY / SESSION MANAGEMENT
# -------------------------------------------------------------------

class SessionState:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.messages: List[Dict[str, Any]] = []  # simple chat history
        self.created_at = datetime.utcnow()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})


class SessionManager:
    """
    Simple in-memory session service.
    For a real deployment, this could be Redis, database, etc.
    """

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_session(self, user_id: str) -> SessionState:
        if user_id not in self._sessions:
            self._sessions[user_id] = SessionState(user_id)
        return self._sessions[user_id]


class LongTermMemory:
    """
    Implements very simple long-term memory using a JSON file.
    Stores a TravelPreferences object per user_id.
    """

    def __init__(self, filepath: str = "user_preferences.json"):
        self.filepath = filepath
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                logger.warning("Failed to load long-term memory file; starting empty.")
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving long-term memory: {e}")

    def get_preferences(self, user_id: str) -> Optional[TravelPreferences]:
        record = self._data.get(user_id)
        if not record:
            return None
        return TravelPreferences(**record)

    def save_preferences(self, user_id: str, prefs: TravelPreferences):
        self._data[user_id] = asdict(prefs)
        self._save()


# -------------------------------------------------------------------
# TOOLS (Custom, Cost-Free)
# -------------------------------------------------------------------

def days_between(start_date: str, end_date: str) -> int:
    s = datetime.fromisoformat(start_date)
    e = datetime.fromisoformat(end_date)
    return max(1, (e - s).days)


def estimate_flight_cost(home_city: str, destination: str, travelers: int, budget_level: str) -> float:
    """
    Very rough heuristic:
    - base_price by 'distance bucket' approximated via dummy mapping.
    - multiplier by budget level.
    """
    # dummy distance mapping
    base_routes = {
        "chennai": {"mumbai": 80, "delhi": 120, "bangkok": 200, "europe": 500},
    }
    home_key = home_city.lower()
    dest_key = destination.lower()

    base_price = 200.0  # fallback
    if home_key in base_routes and dest_key in base_routes[home_key]:
        base_price = base_routes[home_key][dest_key]
    else:
        # if unknown, assume mid-haul
        base_price = 300.0

    budget_mult = {"low": 0.9, "medium": 1.0, "high": 1.3}.get(budget_level, 1.0)
    return base_price * travelers * budget_mult


def estimate_accommodation_cost(
    nights: int,
    travelers: int,
    preferred_accommodation: str,
    budget_level: str,
) -> float:
    """
    Rough heuristic cost per night.
    """
    base_per_night = {
        "hostel": 20,
        "hotel": 60,
        "airbnb": 40,
    }.get(preferred_accommodation.lower(), 50)

    budget_mult = {"low": 0.8, "medium": 1.0, "high": 1.4}.get(budget_level, 1.0)
    cost_per_night = base_per_night * budget_mult * max(1, travelers / 2.0)
    return cost_per_night * max(1, nights)


def estimate_daily_spend(travelers: int, pace: str, interests: List[str], budget_level: str) -> float:
    """
    Very simple: base per person per day, adjusted by pace and interests.
    """
    base_per_person = {"low": 20, "medium": 35, "high": 60}.get(budget_level, 35)
    pace_mult = {"relaxed": 0.9, "packed": 1.2}.get(pace, 1.0)

    excitement_mult = 1.0
    if any(i.lower() in ["nightlife", "adventure", "fine dining"] for i in interests):
        excitement_mult += 0.2

    return base_per_person * travelers * pace_mult * excitement_mult


def suggest_activities_for_destination(destination: str, interests: List[str], days: int) -> Dict[int, List[str]]:
    """
    Offline, dictionary-based suggestions based on destination + interest.
    This is deliberately static so it costs nothing and has no API calls.
    """
    dest = destination.lower()
    db = {
        "bangkok": {
            "culture": ["Visit Grand Palace", "Temple of the Emerald Buddha"],
            "food": ["Street food tour in Chinatown", "Floating market visit"],
            "shopping": ["MBK Center", "Chatuchak Weekend Market"],
            "nightlife": ["Rooftop bar evening", "Night market walk"],
        },
        "mumbai": {
            "culture": ["Gateway of India", "Heritage walk at Fort"],
            "food": ["Street food at Juhu", "Local café hopping"],
            "shopping": ["Colaba Causeway", "Phoenix Mall"],
            "nature": ["Marine Drive sunset", "Sanjay Gandhi National Park"],
        },
        "default": {
            "culture": ["Old town walking tour", "Historical museum"],
            "food": ["Local food market", "Popular restaurant"],
            "shopping": ["Central shopping district", "Local craft market"],
            "nature": ["City park walk", "Scenic viewpoint"],
        },
    }

    base = db.get(dest, db["default"])

    mapped_interest_keys = {
        "culture": ["culture", "history", "heritage", "museums"],
        "food": ["food", "cuisine", "restaurants", "street food"],
        "shopping": ["shopping", "markets", "souvenirs"],
        "nightlife": ["nightlife", "bars", "clubs"],
        "nature": ["nature", "hiking", "parks", "beach"],
    }

    # Choose at most 3 main interest categories
    selected_categories = []
    for category, keywords in mapped_interest_keys.items():
        if any(kw in " ".join(interests).lower() for kw in keywords):
            selected_categories.append(category)
    if not selected_categories:
        selected_categories = ["culture", "food"]

    schedule: Dict[int, List[str]] = {}
    for day in range(1, days + 1):
        category = selected_categories[(day - 1) % len(selected_categories)]
        schedule[day] = base.get(category, [])[:3]  # up to 3 suggestions

    return schedule


def simple_weather_stub(destination: str, days: int) -> Dict[int, str]:
    """
    Weather stub – returns a generic description per day.
    (No external API, pure stub to demonstrate 'tooling'.)
    """
    outlook_cycle = ["sunny", "partly cloudy", "warm", "chance of showers"]
    result = {}
    for day in range(1, days + 1):
        result[day] = outlook_cycle[(day - 1) % len(outlook_cycle)]
    return result


# -------------------------------------------------------------------
# BASE AGENT CLASS
# -------------------------------------------------------------------

class BaseAgent:
    def __init__(self, name: str, llm: LocalLLM):
        self.name = name
        self.llm = llm
        self.logger = logging.getLogger(name)

    def run(self, *args, **kwargs):
        raise NotImplementedError


# -------------------------------------------------------------------
# AGENT 1: PREFERENCE AGENT
# -------------------------------------------------------------------

class PreferenceAgent(BaseAgent):
    """
    Reads existing long-term preferences if present, and if missing,
    builds a basic profile using the current TravelRequest.

    In a more interactive setup, this agent would chat with the user.
    Here we derive a reasonable default from the request plus memory.
    """

    def run(
        self,
        request: TravelRequest,
        long_term_memory: LongTermMemory,
    ) -> TravelPreferences:
        self.logger.info("Running PreferenceAgent...")
        existing = long_term_memory.get_preferences(request.user_id)
        if existing:
            self.logger.info("Found existing preferences in long-term memory.")
            # Optionally update with any new info
            # For simplicity, we keep existing prefs unchanged
            return existing

        # Build new preferences using request hints
        if request.total_budget < 300:
            level = "low"
        elif request.total_budget < 1000:
            level = "medium"
        else:
            level = "high"

        prefs = TravelPreferences(
            home_city="Chennai",  # you can parameterize this in CLI
            typical_budget_level=level,
            preferred_accommodation="hotel",
            pace=request.style or "relaxed",
            top_interests=request.interests[:3],
        )
        long_term_memory.save_preferences(request.user_id, prefs)
        self.logger.info("Created new preferences and stored in long-term memory.")
        return prefs


# -------------------------------------------------------------------
# AGENT 2: BUDGET AGENT
# -------------------------------------------------------------------

class BudgetAgent(BaseAgent):
    """
    Uses tools to estimate flight + accommodation + daily spend, then
    checks if the plan is within budget.
    """

    def run(
        self,
        request: TravelRequest,
        preferences: TravelPreferences,
    ) -> BudgetSummary:
        self.logger.info("Running BudgetAgent...")
        n_days = days_between(request.start_date, request.end_date)
        n_nights = max(1, n_days - 1)

        flight_cost = estimate_flight_cost(
            home_city=preferences.home_city,
            destination=request.destination,
            travelers=request.travelers,
            budget_level=preferences.typical_budget_level,
        )

        accommodation_cost = estimate_accommodation_cost(
            nights=n_nights,
            travelers=request.travelers,
            preferred_accommodation=preferences.preferred_accommodation,
            budget_level=preferences.typical_budget_level,
        )

        daily_spend = estimate_daily_spend(
            travelers=request.travelers,
            pace=preferences.pace,
            interests=request.interests,
            budget_level=preferences.typical_budget_level,
        )

        total_estimated = flight_cost + accommodation_cost + daily_spend * n_days
        within = total_estimated <= request.total_budget

        if within:
            notes = (
                f"Estimated total {total_estimated:.0f} fits within your budget "
                f"of {request.total_budget:.0f}."
            )
        else:
            notes = (
                f"Estimated total {total_estimated:.0f} exceeds your budget "
                f"of {request.total_budget:.0f}. Consider adjusting dates, "
                f"destination, or accommodation type."
            )

        self.logger.info(notes)

        return BudgetSummary(
            estimated_flight_cost=flight_cost,
            estimated_accommodation_cost=accommodation_cost,
            estimated_daily_spend=daily_spend,
            total_estimated_cost=total_estimated,
            is_within_budget=within,
            notes=notes,
        )


# -------------------------------------------------------------------
# AGENT 3: ITINERARY AGENT
# -------------------------------------------------------------------

class ItineraryAgent(BaseAgent):
    """
    Uses LocalLLM plus custom tools for activities + weather stub
    to create a basic, human-readable day-by-day itinerary.
    """

    def run(
        self,
        request: TravelRequest,
        preferences: TravelPreferences,
        budget: BudgetSummary,
    ) -> ItineraryResult:
        self.logger.info("Running ItineraryAgent...")
        n_days = days_between(request.start_date, request.end_date)

        activity_plan = suggest_activities_for_destination(
            destination=request.destination,
            interests=request.interests,
            days=n_days,
        )
        weather_plan = simple_weather_stub(request.destination, n_days)

        # Compose a prompt-like string to feed into our stub LLM
        system_prompt = (
            "You are a friendly travel planner who writes concise itineraries.\n"
            "Use the suggested activities but feel free to reorder them logically.\n"
        )
        description = (
            f"Destination: {request.destination}\n"
            f"Dates: {request.start_date} to {request.end_date} ({n_days} days)\n"
            f"Budget note: {budget.notes}\n"
            f"Pace: {preferences.pace}\n"
            f"Interests: {', '.join(request.interests)}\n"
            "Suggested activities per day (draft):\n"
        )
        for day in range(1, n_days + 1):
            acts = activity_plan.get(day, [])
            w = weather_plan.get(day, "pleasant")
            description += f"  Day {day} (likely {w}): {', '.join(acts) or 'Free exploration'}\n"

        user_prompt = (
            description
            + "\n\nPlease turn this into a readable, day-by-day itinerary."
        )

        itinerary_text = self.llm.generate(system_prompt, user_prompt, max_tokens=512)

        # Split into daily chunks (very naive).
        daily_plan: Dict[int, str] = {}
        lines = [l.strip() for l in itinerary_text.splitlines() if l.strip()]
        current_day = 1
        current_text: List[str] = []
        for line in lines:
            if line.lower().startswith("day "):
                if current_text:
                    daily_plan[current_day] = "\n".join(current_text)
                    current_day += 1
                    current_text = []
            current_text.append(line)
        if current_text:
            daily_plan[current_day] = "\n".join(current_text)

        return ItineraryResult(
            raw_text=itinerary_text,
            daily_plan=daily_plan,
        )


# -------------------------------------------------------------------
# AGENT 4: EVALUATION AGENT
# -------------------------------------------------------------------

class EvaluationAgent(BaseAgent):
    """
    Simple heuristic agent that 'scores' the itinerary using:
      - budget fit
      - interest coverage
      - a short LLM-based qualitative comment (stub)
    """

    def run(
        self,
        request: TravelRequest,
        preferences: TravelPreferences,
        budget: BudgetSummary,
        itinerary: ItineraryResult,
    ) -> EvaluationResult:
        self.logger.info("Running EvaluationAgent...")

        # Budget score
        if budget.is_within_budget:
            score_budget = 90
        else:
            # degrade proportional to overshoot
            overshoot_ratio = max(
                0.0, (budget.total_estimated_cost - request.total_budget) / request.total_budget
            )
            score_budget = max(20.0, 90.0 * (1 - overshoot_ratio))

        # Interest alignment: count how many interests appear in itinerary text
        text_lower = itinerary.raw_text.lower()
        hit_count = sum(1 for i in request.interests if i.lower() in text_lower)
        if not request.interests:
            score_interests = 70
        else:
            ratio = hit_count / len(request.interests)
            score_interests = 40 + ratio * 60  # 40 to 100

        # overall as a simple average
        score_overall = (score_budget + score_interests) / 2

        system_prompt = "You are evaluating a travel itinerary for quality and fit."
        user_prompt = (
            "Evaluate the following itinerary for the user and give high-level feedback.\n\n"
            f"User interests: {', '.join(request.interests)}\n"
            f"Budget note: {budget.notes}\n\n"
            f"Itinerary:\n{itinerary.raw_text}\n\n"
            "Provide a short paragraph of feedback."
        )
        comment = self.llm.generate(system_prompt, user_prompt, max_tokens=256)

        return EvaluationResult(
            score_overall=round(score_overall, 1),
            score_budget_fit=round(score_budget, 1),
            score_interest_alignment=round(score_interests, 1),
            comments=comment,
        )


# -------------------------------------------------------------------
# ORCHESTRATOR (MULTI-AGENT PIPELINE + METRICS)
# -------------------------------------------------------------------

class TravelConciergeOrchestrator:
    """
    High-level agent that coordinates the four sub-agents
    and exposes a single `plan_trip` method.

    This is what you'd call from a CLI, web app, or notebook.
    """

    def __init__(self, llm: Optional[LocalLLM] = None):
        self.llm = llm or LocalLLM()
        self.session_manager = SessionManager()
        self.long_term_memory = LongTermMemory()

        self.preference_agent = PreferenceAgent("PreferenceAgent", self.llm)
        self.budget_agent = BudgetAgent("BudgetAgent", self.llm)
        self.itinerary_agent = ItineraryAgent("ItineraryAgent", self.llm)
        self.evaluation_agent = EvaluationAgent("EvaluationAgent", self.llm)

    def plan_trip(self, request: TravelRequest) -> Dict[str, Any]:
        session = self.session_manager.get_session(request.user_id)
        session.add_message("user", f"Plan a trip to {request.destination}")

        timings: Dict[str, float] = {}

        # Agent 1
        t0 = time.time()
        prefs = self.preference_agent.run(request, self.long_term_memory)
        timings["PreferenceAgent"] = time.time() - t0

        # Agent 2
        t0 = time.time()
        budget = self.budget_agent.run(request, prefs)
        timings["BudgetAgent"] = time.time() - t0

        # Agent 3
        t0 = time.time()
        itinerary = self.itinerary_agent.run(request, prefs, budget)
        timings["ItineraryAgent"] = time.time() - t0

        # Agent 4
        t0 = time.time()
        evaluation = self.evaluation_agent.run(request, prefs, budget, itinerary)
        timings["EvaluationAgent"] = time.time() - t0

        # Aggregate result
        result = {
            "request": asdict(request),
            "preferences": asdict(prefs),
            "budget_summary": asdict(budget),
            "itinerary": {
                "raw_text": itinerary.raw_text,
                "daily_plan": itinerary.daily_plan,
            },
            "evaluation": {
                "score_overall": evaluation.score_overall,
                "score_budget_fit": evaluation.score_budget_fit,
                "score_interest_alignment": evaluation.score_interest_alignment,
                "comments": evaluation.comments,
            },
            "timings_seconds": timings,
        }

        # Basic observability log
        logger.info("Trip planning complete.")
        logger.info(f"Overall score: {evaluation.score_overall}")
        logger.info(f"Timings (s): {timings}")

        return result


# -------------------------------------------------------------------
# SIMPLE CLI INTERFACE (FOR DEMO / LOCAL USE)
# -------------------------------------------------------------------

def prompt_user_for_request(user_id: str) -> TravelRequest:
    print("=== Travel Concierge Agent ===")
    destination = input("Destination (e.g., Bangkok, Mumbai, Europe): ").strip() or "Bangkok"
    start_date = input("Start date (YYYY-MM-DD): ").strip() or datetime.today().strftime("%Y-%m-%d")
    end_date_default = (datetime.today() + timedelta(days=4)).strftime("%Y-%m-%d")
    end_date = input(f"End date (YYYY-MM-DD) [default {end_date_default}]: ").strip() or end_date_default

    budget_str = input("Total budget in your currency (e.g., 800): ").strip() or "800"
    try:
        total_budget = float(budget_str)
    except ValueError:
        total_budget = 800.0

    travelers_str = input("Number of travelers [default 2]: ").strip() or "2"
    try:
        travelers = int(travelers_str)
    except ValueError:
        travelers = 2

    style = input("Travel style (relaxed/packed/family/adventure) [relaxed]: ").strip() or "relaxed"

    interests_str = input(
        "List a few interests separated by commas (e.g., food, culture, shopping): "
    ).strip()
    interests = [i.strip() for i in interests_str.split(",") if i.strip()] or ["food", "culture"]

    return TravelRequest(
        user_id=user_id,
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        total_budget=total_budget,
        travelers=travelers,
        style=style,
        interests=interests,
    )


def pretty_print_plan(plan: Dict[str, Any]):
    print("\n==============================")
    print(" TRAVEL CONCIERGE – SUMMARY")
    print("==============================\n")

    req = plan["request"]
    prefs = plan["preferences"]
    budget = plan["budget_summary"]
    itinerary = plan["itinerary"]
    evaluation = plan["evaluation"]
    timings = plan["timings_seconds"]

    print("Request:")
    print(f"  Destination: {req['destination']}")
    print(f"  Dates      : {req['start_date']} to {req['end_date']}")
    print(f"  Travelers  : {req['travelers']}")
    print(f"  Budget     : {req['total_budget']:.0f}")
    print(f"  Style      : {req['style']}")
    print(f"  Interests  : {', '.join(req['interests'])}")
    print()

    print("Preferences (from long-term memory):")
    print(f"  Home city            : {prefs['home_city']}")
    print(f"  Typical budget level : {prefs['typical_budget_level']}")
    print(f"  Preferred stay       : {prefs['preferred_accommodation']}")
    print(f"  Pace                 : {prefs['pace']}")
    print(f"  Top interests        : {', '.join(prefs['top_interests'])}")
    print()

    print("Budget Summary:")
    print(f"  Flight cost estimate         : {budget['estimated_flight_cost']:.0f}")
    print(f"  Accommodation cost estimate  : {budget['estimated_accommodation_cost']:.0f}")
    print(f"  Daily spend estimate (per all travelers): {budget['estimated_daily_spend']:.0f}")
    print(f"  Total estimated cost         : {budget['total_estimated_cost']:.0f}")
    print(f"  Within budget?               : {'Yes' if budget['is_within_budget'] else 'No'}")
    print(f"  Notes                        : {budget['notes']}")
    print()

    print("Itinerary (raw):")
    print(itinerary["raw_text"])
    print()

    print("Evaluation:")
    print(f"  Overall score          : {evaluation['score_overall']}")
    print(f"  Budget fit score       : {evaluation['score_budget_fit']}")
    print(f"  Interest alignment     : {evaluation['score_interest_alignment']}")
    print("  Comments:")
    print(f"    {evaluation['comments']}")
    print()

    print("Agent timings (seconds):")
    for agent_name, t in timings.items():
        print(f"  {agent_name}: {t:.3f}s")
    print()


# -------------------------------------------------------------------
# MAIN ENTRYPOINT
# -------------------------------------------------------------------

def main():
    user_id = "demo_user"  # In a real app, this would be the authenticated user ID.
    request = prompt_user_for_request(user_id=user_id)
    orchestrator = TravelConciergeOrchestrator()
    plan = orchestrator.plan_trip(request)
    pretty_print_plan(plan)


if __name__ == "__main__":
    main()
