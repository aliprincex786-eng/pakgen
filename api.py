import ast
import math
import operator
import os
import re
from typing import Union

import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "AI Calculator"
GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧮",
    layout="centered",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 2rem;
        }

        .result-box {
            padding: 1.5rem;
            border-radius: 16px;
            background: linear-gradient(
                135deg,
                #eef2ff,
                #f8fafc
            );
            border: 1px solid #e5e7eb;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .result-label {
            color: #6b7280;
            font-size: 0.9rem;
        }

        .result-value {
            font-size: 2.4rem;
            font-weight: 800;
            color: #4f46e5;
        }

        .history-item {
            padding: 0.75rem;
            border-bottom: 1px solid #e5e7eb;
        }

        .example-box {
            padding: 1rem;
            border-radius: 10px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str:
    """
    Read Gemini API key from Streamlit secrets first,
    then environment variables.
    """

    try:
        key = st.secrets.get("GEMINI_API_KEY")

        if key:
            return key
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "")


# ============================================================
# SAFE MATHEMATICAL CALCULATOR
# ============================================================

# Allowed mathematical operators.
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# Allowed mathematical functions.
SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def safe_eval(node: ast.AST) -> Union[int, float]:
    """
    Safely evaluate a mathematical AST.

    This intentionally does NOT use Python's eval().
    """

    if isinstance(node, ast.Expression):
        return safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numeric values are allowed.")

    if isinstance(node, ast.BinOp):
        operation = BINARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported mathematical operator.")

        left = safe_eval(node.left)
        right = safe_eval(node.right)

        # Prevent extremely large exponent calculations.
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError(
                    "Exponent is too large."
                )

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported unary operator.")

        return operation(safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function.")

        function_name = node.func.id

        if function_name not in SAFE_FUNCTIONS:
            raise ValueError(
                f"Function '{function_name}' is not allowed."
            )

        function = SAFE_FUNCTIONS[function_name]

        arguments = [
            safe_eval(argument)
            for argument in node.args
        ]

        return function(*arguments)

    if isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]

        raise ValueError(
            f"Unknown constant '{node.id}'."
        )

    raise ValueError(
        "This expression contains unsupported syntax."
    )


def calculate_expression(expression: str) -> float:
    """
    Calculate a mathematical expression safely.
    """

    expression = expression.strip()

    if not expression:
        raise ValueError(
            "Please enter a calculation."
        )

    # Basic input-length protection.
    if len(expression) > 500:
        raise ValueError(
            "Calculation is too long."
        )

    # Allow common visual multiplication symbol.
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")

    # Convert simple percentage syntax:
    # 15% -> 15/100
    expression = re.sub(
        r"(\d+(?:\.\d+)?)\s*%",
        r"(\1/100)",
        expression,
    )

    tree = ast.parse(
        expression,
        mode="eval",
    )

    result = safe_eval(tree)

    if not math.isfinite(result):
        raise ValueError(
            "The calculation produced an invalid result."
        )

    return result


def format_number(value: float) -> str:
    """
    Make calculator output easier to read.
    """

    if isinstance(value, int):
        return str(value)

    if value == int(value):
        return f"{int(value):,}"

    return f"{value:,.10g}"


# ============================================================
# GEMINI AI
# ============================================================

def ask_gemini(user_question: str) -> dict:
    """
    Ask Gemini to solve/explain a natural-language calculation.
    """

    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "to Streamlit secrets."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are an accurate mathematical assistant.

Solve the user's calculation/problem.

USER QUESTION:
{user_question}

Rules:
1. Give the final numerical answer.
2. Show the important calculation steps.
3. Keep the explanation easy to understand.
4. Do not invent missing information.
5. If the question is ambiguous, clearly explain what information is missing.
6. Double-check arithmetic before answering.
7. Do not give financial, medical, legal, or investment advice.
   If the user asks for such advice, only perform the requested
   mathematical calculation and clearly state that the calculation
   is not professional advice.

Return JSON with exactly these fields:

{{
  "answer": "Final answer",
  "steps": [
    "Step 1",
    "Step 2"
  ],
  "explanation": "Short explanation"
}}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    import json

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned an invalid response."
        ) from exc


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧮 AI Calculator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Calculate numbers instantly or ask Gemini to solve
        a calculation in natural language.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Calculator")

    st.markdown(
        """
        ### Examples

        - `25 * 18`
        - `5000 / 12`
        - `15% of 80000`
        - `sqrt(144)`
        - `2^10` is supported as `2**10`
        - `What is 18% of 75000?`
        - `If I save 20% of 90000, how much is that?`
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear History",
        use_container_width=True,
    ):
        st.session_state.history = []
        st.rerun()


# ============================================================
# INPUT
# ============================================================

question = st.text_input(
    "Enter your calculation",
    placeholder=(
        "Example: What is 18% of 75000?"
    ),
)

calculate_button = st.button(
    "🧮 Calculate",
    type="primary",
    use_container_width=True,
)


# ============================================================
# CALCULATION
# ============================================================

if calculate_button:

    if not question.strip():
        st.warning(
            "Please enter a calculation."
        )
        st.stop()

    # --------------------------------------------------------
    # First attempt: direct Python calculation
    # --------------------------------------------------------

    direct_expression = question.strip()

    # Handle common natural-language percentage expressions.
    percentage_match = re.fullmatch(
        r"""
        \s*
        (?:what\s+is\s+)?
        (\d+(?:\.\d+)?)\s*%
        \s+of\s+
        (\d+(?:\.\d+)?)
        \s*\??\s*
        """,
        direct_expression,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    try:

        if percentage_match:

            percentage = float(
                percentage_match.group(1)
            )

            number = float(
                percentage_match.group(2)
            )

            result = (
                percentage / 100
            ) * number

            formatted_result = format_number(
                result
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Result</div>
                    <div class="result-value">
                        {formatted_result}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                f"{percentage}% of {format_number(number)} "
                f"= **{formatted_result}**"
            )

            st.session_state.history.insert(
                0,
                {
                    "question": question,
                    "answer": formatted_result,
                    "method": "Python calculator",
                },
            )

        else:

            # Try direct mathematical expression.
            result = calculate_expression(
                direct_expression
            )

            formatted_result = format_number(
                result
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Result</div>
                    <div class="result-value">
                        {formatted_result}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.code(
                f"{question} = {formatted_result}"
            )

            st.session_state.history.insert(
                0,
                {
                    "question": question,
                    "answer": formatted_result,
                    "method": "Python calculator",
                },
            )

    except Exception:

        # ----------------------------------------------------
        # If it isn't a normal expression, use Gemini.
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Gemini is solving your calculation..."
            ):

                ai_result = ask_gemini(
                    question
                )

            answer = ai_result.get(
                "answer",
                "No answer returned.",
            )

            steps = ai_result.get(
                "steps",
                [],
            )

            explanation = ai_result.get(
                "explanation",
                "",
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">
                        Gemini Result
                    </div>
                    <div class="result-value">
                        {answer}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if steps:

                st.subheader(
                    "📐 Calculation Steps"
                )

                for index, step in enumerate(
                    steps,
                    start=1,
                ):
                    st.markdown(
                        f"{index}. {step}"
                    )

            if explanation:

                st.subheader(
                    "💡 Explanation"
                )

                st.write(
                    explanation
                )

            st.session_state.history.insert(
                0,
                {
                    "question": question,
                    "answer": answer,
                    "method": "Gemini",
                },
            )

        except Exception as exc:

            st.error(
                "I couldn't calculate that."
            )

            with st.expander(
                "Technical details"
            ):
                st.exception(exc)


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "🕘 Calculation History"
    )

    for item in st.session_state.history[:10]:

        st.markdown(
            f"""
            <div class="history-item">
                <strong>
                    {item["question"]}
                </strong>
                <br>
                = {item["answer"]}
                <br>
                <small>
                    Method: {item["method"]}
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Calculator • Python handles direct arithmetic; "
    "Gemini handles natural-language calculation problems."
)
