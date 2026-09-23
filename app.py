"""
Smart Traffic Signal Timing Controller
Academic Internal Assessment Project

Tech stack:
- Streamlit: user interface
- LangChain + Groq ChatGroq: natural-language traffic scenario parsing
- scikit-fuzzy: genuine fuzzy inference system
- NumPy: numerical operations

Run:
    streamlit run app.py

Environment:
    GROQ_API_KEY=<your xAI API key>

For Streamlit Community Cloud, add GROQ_API_KEY under:
    App -> Settings -> Secrets
"""

import json
import os
import re
from typing import Any, Dict, Tuple
from dotenv import load_dotenv
import numpy as np
import streamlit as st

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError as exc:
    st.error(
        "scikit-fuzzy is not installed. Install dependencies with: "
        "pip install -r requirements.txt"
    )
    st.stop()

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Traffic Signal Timing Controller",
    page_icon="🚦",
    layout="wide",
)





def get_groq_key() -> str:
    """Read the Groq API key from the environment."""
    return os.getenv("GROQ_API_KEY", "").strip()


@st.cache_resource(show_spinner=False)
def create_llm() -> Any:
    """Create and cache the LangChain Groq chat model."""
    api_key = get_groq_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Make sure it exists in your .env file."
        )

    if ChatGroq is None:
        raise RuntimeError(
            "langchain-groq is not installed. Run: "
            "pip install langchain-groq"
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object even if the LLM wraps it in markdown."""
    text = text.strip()

    # First try the complete response.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Then find the first JSON-looking object.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError("The AI response did not contain valid JSON.")


def clamp_metric(value: Any, low: float, high: float) -> float:
    """Convert a model-produced number into a safe bounded float."""
    try:
        return float(np.clip(float(value), low, high))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid metric value: {value!r}")


def parse_traffic_scenario(scenario: str) -> Tuple[float, float]:
    """
    Use Groq through LangChain to extract:
      - vehicle_density: 0-10
      - waiting_time: 0-10 minutes

    The response is deliberately constrained to JSON so the fuzzy controller
    receives numeric values rather than free-form prose.
    """
    llm = create_llm()

    prompt = f"""
You are the traffic-scenario parser for an academic Smart Traffic Signal
Timing Controller.

Read the scenario below and extract ONLY these two numerical metrics:

1. vehicle_density: traffic/vehicle density on a 0-10 scale.
   0 = almost no vehicles
   10 = extremely congested

2. waiting_time: approximate vehicle waiting time in MINUTES on a 0-10 scale.
   0 = no meaningful wait
   10 = 10 minutes or more

Rules:
- Infer reasonable values from natural-language descriptions.
- If a value is not explicitly stated, estimate it conservatively from context.
- Keep both values within their stated ranges.
- Do not include explanations.
- Return ONLY valid JSON with exactly these keys:
  {{"vehicle_density": number, "waiting_time": number}}

Scenario:
{scenario}
""".strip()

    response = llm.invoke(prompt)
    content = response.content

    # Chat models normally return a string, but handle content blocks safely.
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        content = "".join(parts)

    data = extract_json_object(str(content))

    density = clamp_metric(data.get("vehicle_density"), 0, 10)
    waiting = clamp_metric(data.get("waiting_time"), 0, 10)

    return density, waiting


def generate_explanation(
    vehicle_density: float,
    waiting_time: float,
    green_duration: float,
    source: str,
) -> str:
    """Ask Groq for a controller-facing explanation of the fuzzy result."""
    llm = create_llm()

    prompt = f"""
You are explaining the result of an academic fuzzy-logic traffic signal
controller to a traffic controller.

Inputs:
- Vehicle density: {vehicle_density:.1f}/10
- Waiting time: {waiting_time:.1f} minutes
- Calculated green-light duration: {green_duration:.1f} seconds
- Input source: {source}

Write a short, friendly, professional explanation in 2-4 sentences.
Explain how density and waiting time affected the green-light duration.
Do not claim that this is a real-world certified traffic-control system.
Do not invent sensor measurements or safety guarantees.
""".strip()

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        content = "".join(parts)

    return str(content).strip()


# ---------------------------------------------------------------------------
# Fuzzy Inference System
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_fuzzy_controller():
    """
    Build a genuine Mamdani-style fuzzy inference system.

    Inputs:
        Vehicle Density: 0-10
        Waiting Time:    0-10 minutes

    Output:
        Green Light Duration: 10-90 seconds

    Defuzzification:
        Centroid
    """
    # Antecedents
    vehicle_density = ctrl.Antecedent(
        np.linspace(0, 10, 101),
        "vehicle_density",
    )
    waiting_time = ctrl.Antecedent(
        np.linspace(0, 10, 101),
        "waiting_time",
    )

    # Consequent
    green_duration = ctrl.Consequent(
        np.linspace(10, 90, 801),
        "green_duration",
        defuzzify_method="centroid",
    )

    # Membership functions for vehicle density.
    vehicle_density["low"] = fuzz.trimf(
        vehicle_density.universe, [0, 0, 5]
    )
    vehicle_density["medium"] = fuzz.trimf(
        vehicle_density.universe, [2, 5, 8]
    )
    vehicle_density["high"] = fuzz.trimf(
        vehicle_density.universe, [5, 10, 10]
    )

    # Membership functions for waiting time.
    waiting_time["short"] = fuzz.trimf(
        waiting_time.universe, [0, 0, 4]
    )
    waiting_time["medium"] = fuzz.trimf(
        waiting_time.universe, [2, 5, 8]
    )
    waiting_time["long"] = fuzz.trimf(
        waiting_time.universe, [5, 10, 10]
    )

    # Output membership functions.
    green_duration["short"] = fuzz.trimf(
        green_duration.universe, [10, 10, 40]
    )
    green_duration["medium"] = fuzz.trimf(
        green_duration.universe, [25, 50, 75]
    )
    green_duration["long"] = fuzz.trimf(
        green_duration.universe, [60, 90, 90]
    )

    # -----------------------------------------------------------------------
    # Rule base
    # -----------------------------------------------------------------------
    # The rule base increases green time as congestion and/or waiting grows.
    rules = [
        ctrl.Rule(
            vehicle_density["low"] & waiting_time["short"],
            green_duration["short"],
            label="R1: low density + short wait -> short green",
        ),
        ctrl.Rule(
            vehicle_density["low"] & waiting_time["medium"],
            green_duration["medium"],
            label="R2: low density + medium wait -> medium green",
        ),
        ctrl.Rule(
            vehicle_density["low"] & waiting_time["long"],
            green_duration["long"],
            label="R3: low density + long wait -> long green",
        ),
        ctrl.Rule(
            vehicle_density["medium"] & waiting_time["short"],
            green_duration["medium"],
            label="R4: medium density + short wait -> medium green",
        ),
        ctrl.Rule(
            vehicle_density["medium"] & waiting_time["medium"],
            green_duration["medium"],
            label="R5: medium density + medium wait -> medium green",
        ),
        ctrl.Rule(
            vehicle_density["medium"] & waiting_time["long"],
            green_duration["long"],
            label="R6: medium density + long wait -> long green",
        ),
        ctrl.Rule(
            vehicle_density["high"] & waiting_time["short"],
            green_duration["long"],
            label="R7: high density + short wait -> long green",
        ),
        ctrl.Rule(
            vehicle_density["high"] & waiting_time["medium"],
            green_duration["long"],
            label="R8: high density + medium wait -> long green",
        ),
        ctrl.Rule(
            vehicle_density["high"] & waiting_time["long"],
            green_duration["long"],
            label="R9: high density + long wait -> long green",
        ),
    ]

    system = ctrl.ControlSystem(rules)
    return system, vehicle_density, waiting_time, green_duration


def run_fuzzy_controller(
    density: float,
    waiting: float,
) -> float:
    """Run fuzzy rule evaluation and centroid defuzzification."""
    system, _, _, _ = build_fuzzy_controller()
    simulation = ctrl.ControlSystemSimulation(system)

    simulation.input["vehicle_density"] = float(density)
    simulation.input["waiting_time"] = float(waiting)

    simulation.compute()

    result = float(simulation.output["green_duration"])
    return float(np.clip(result, 10, 90))


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("🚦 Smart Traffic Signal Timing Controller")
st.caption(
    "Natural-language traffic analysis → Groq/LangChain → "
    "Fuzzy Inference System → Green-light timing"
)

st.info(
    "Academic demonstration: the application combines an LLM for extracting "
    "traffic metrics with a Mamdani fuzzy controller for the final timing."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Controller Settings")
    st.write("**LLM:** `Groq / Llama 3.3 70B`")
    st.write("**Fuzzy method:** Mamdani-style inference")
    st.write("**Defuzzification:** Centroid")
    st.write("**Output range:** 10–90 seconds")

    st.divider()
    st.subheader("Fuzzy Input Ranges")
    st.write("Vehicle density: 0–10")
    st.write("Waiting time: 0–10 minutes")

    st.divider()
    st.subheader("API Key")
    if get_groq_key():
        st.success("GROQ_API_KEY detected")
    else:
        st.warning("GROQ_API_KEY not detected")


scenario = st.text_area(
    "📝 Describe the traffic scenario",
    value=(
        "Heavy traffic jam on Lane 1 with vehicles waiting for over "
        "4 minutes, no emergency."
    ),
    height=120,
    help="Describe the traffic situation naturally. Groq will extract density and waiting time.",
)

st.subheader("Manual Override")

override_enabled = st.checkbox(
    "Use manual slider values instead of the AI-extracted values",
    value=False,
)

col1, col2 = st.columns(2)

with col1:
    manual_density = st.slider(
        "Vehicle Density (0–10)",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        disabled=not override_enabled,
    )

with col2:
    manual_waiting = st.slider(
        "Waiting Time (0–10 minutes)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
        disabled=not override_enabled,
    )

calculate = st.button(
    "🚦 Calculate Green-Light Duration",
    type="primary",
    use_container_width=True,
)

if calculate:
    if not scenario.strip():
        st.error("Please enter a traffic scenario.")
        st.stop()

    # ---------------------------------------------------------------
    # Step 1: obtain input metrics
    # ---------------------------------------------------------------
    if override_enabled:
        density = float(manual_density)
        waiting = float(manual_waiting)
        source = "manual slider override"
        ai_parse_message = (
            "Manual override is enabled, so the sliders were used directly."
        )
    else:
        if not get_groq_key():
            st.error(
                "GROQ_API_KEY is required for natural-language parsing. "
                "Add it to your environment or Streamlit Secrets, or enable "
                "Manual Override and use the sliders."
            )
            st.stop()

        try:
            with st.spinner("Groq is interpreting the traffic scenario..."):
                density, waiting = parse_traffic_scenario(scenario)

            source = "Groq natural-language extraction"
            ai_parse_message = (
                "Groq extracted the traffic metrics from the scenario."
            )
        except Exception as exc:
            st.error(f"AI parsing failed: {exc}")
            st.stop()

    # ---------------------------------------------------------------
    # Step 2: run the fuzzy controller
    # ---------------------------------------------------------------
    try:
        with st.spinner("Evaluating fuzzy rules..."):
            green_duration = run_fuzzy_controller(density, waiting)
    except Exception as exc:
        st.error(f"Fuzzy controller failed: {exc}")
        st.stop()

    # ---------------------------------------------------------------
    # Step 3: display metrics and fuzzy result
    # ---------------------------------------------------------------
    st.divider()
    st.subheader("📊 Controller Inputs")

    m1, m2, m3 = st.columns(3)
    m1.metric("Vehicle Density", f"{density:.1f} / 10")
    m2.metric("Waiting Time", f"{waiting:.1f} min")
    m3.metric("Green Light", f"{green_duration:.1f} sec")

    st.caption(ai_parse_message)

    # Visual progress bars make the extracted/overridden inputs easy to inspect.
    st.write("Vehicle Density")
    st.progress(int(round(density * 10)))

    st.write("Waiting Time")
    st.progress(int(round(waiting * 10)))

    st.success(
        f"Recommended green-light duration: **{green_duration:.1f} seconds**"
    )

    # ---------------------------------------------------------------
    # Step 4: conversational Groq explanation
    # ---------------------------------------------------------------
    if get_groq_key():
        try:
            with st.spinner("Generating controller explanation..."):
                explanation = generate_explanation(
                    density,
                    waiting,
                    green_duration,
                    source,
                )

            st.subheader("💬 AI Explanation")
            st.write(explanation)
        except Exception as exc:
            st.warning(
                "The fuzzy result was calculated successfully, but the "
                f"conversational explanation could not be generated: {exc}"
            )
    else:
        st.info(
            "Add GROQ_API_KEY to enable the conversational explanation."
        )

    # ---------------------------------------------------------------
    # Step 5: show the active fuzzy membership values
    # ---------------------------------------------------------------
    st.subheader("🧠 Fuzzy Membership Analysis")

    system, density_var, waiting_var, duration_var = build_fuzzy_controller()

    density_memberships = {
        label.title(): float(fuzz.interp_membership(
            density_var.universe,
            density_var[label].mf,
            density,
        ))
        for label in ("low", "medium", "high")
    }

    waiting_memberships = {
        label.title(): float(fuzz.interp_membership(
            waiting_var.universe,
            waiting_var[label].mf,
            waiting,
        ))
        for label in ("short", "medium", "long")
    }

    f1, f2 = st.columns(2)

    with f1:
        st.write("**Vehicle Density memberships**")
        for label, value in density_memberships.items():
            st.write(f"{label}: {value:.3f}")
            st.progress(int(round(value * 100)))

    with f2:
        st.write("**Waiting Time memberships**")
        for label, value in waiting_memberships.items():
            st.write(f"{label}: {value:.3f}")
            st.progress(int(round(value * 100)))

    st.caption(
        "The final green-light value is obtained from the activated fuzzy "
        "rules and centroid defuzzification over the 10–90 second output universe."
    )

# ---------------------------------------------------------------------------
# Academic documentation section
# ---------------------------------------------------------------------------
with st.expander("📚 How the system works"):
    st.markdown(
        """
### Processing pipeline

1. **Natural-language input**  
   The user describes a traffic condition in ordinary language.

2. **LLM extraction using LangChain + Groq**  
   Groq extracts:
   - `vehicle_density` → 0 to 10
   - `waiting_time` → 0 to 10 minutes

3. **Fuzzification**  
   The numeric values are converted into fuzzy membership degrees:
   - Vehicle Density → Low / Medium / High
   - Waiting Time → Short / Medium / Long

4. **Fuzzy rule evaluation**  
   A 9-rule Mamdani-style rule base combines the two inputs.

5. **Defuzzification**  
   The activated output membership functions are converted into one crisp
   green-light duration using **centroid defuzzification**.

6. **Conversational explanation**  
   Groq explains the calculated timing in controller-friendly language.

### Example rule

> IF vehicle density is **High** AND waiting time is **Long**  
> THEN green-light duration is **Long**.

### Important separation of responsibilities

The LLM does **not** directly decide the final signal duration. It only
converts the natural-language scenario into numerical input metrics. The
fuzzy inference system performs the actual timing calculation.
"""
    )

with st.expander("📋 Fuzzy Rule Base"):
    rule_rows = [
        ("R1", "Low", "Short", "Short"),
        ("R2", "Low", "Medium", "Medium"),
        ("R3", "Low", "Long", "Long"),
        ("R4", "Medium", "Short", "Medium"),
        ("R5", "Medium", "Medium", "Medium"),
        ("R6", "Medium", "Long", "Long"),
        ("R7", "High", "Short", "Long"),
        ("R8", "High", "Medium", "Long"),
        ("R9", "High", "Long", "Long"),
    ]

    st.table(
        {
            "Rule": [r[0] for r in rule_rows],
            "Vehicle Density": [r[1] for r in rule_rows],
            "Waiting Time": [r[2] for r in rule_rows],
            "Green Duration": [r[3] for r in rule_rows],
        }
    )

st.divider()
st.caption(
    "Smart Traffic Signal Timing Controller • Academic Internal Assessment"
)
# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------