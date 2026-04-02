"""
Calculator tool for mathematical calculations.
The purpose of this tool is for testing.
"""

import ast
import logging
import operator
from typing import Any

from app.agents.tools.agent_tool import AgentTool
from app.repositories.models.custom_bot import BotModel
from app.routes.schemas.conversation import type_model_name
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _ast_eval(node: ast.expr) -> float:
    """Recursively evaluate an AST node using only safe arithmetic operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_ast_eval(node.left), _ast_eval(node.right))
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_ast_eval(node.operand))
    else:
        raise ValueError("Unsupported operation in expression")


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Mathematical expression to evaluate (e.g., '2+2', '10*5', '100/4')"
    )


def calculate_expression(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        str: Result of the calculation or error message
    """
    logger.info(f"[CALCULATOR_TOOL] Calculating expression: {expression}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _ast_eval(tree.body)
        logger.debug(f"[CALCULATOR_TOOL] Calculation result: {result}")

        # Format the result
        if isinstance(result, float) and result.is_integer():
            formatted_result = str(int(result))
        else:
            formatted_result = str(result)

        logger.debug(f"[CALCULATOR_TOOL] Formatted result: {formatted_result}")
        return formatted_result

    except ZeroDivisionError:
        logger.error(f"[CALCULATOR_TOOL] Division by zero in expression: {expression}")
        return "Error: Division by zero is not allowed."
    except (SyntaxError, ValueError) as e:
        logger.warning(f"[CALCULATOR_TOOL] Invalid expression '{expression}': {e}")
        return "Error: Invalid expression. Only basic arithmetic (+, -, *, /) and parentheses are supported."
    except Exception as e:
        logger.error(
            f"[CALCULATOR_TOOL] Error calculating expression '{expression}': {e}"
        )
        return "Error: Unable to calculate the expression. Please check the syntax."


def _calculator_function(
    input_data: CalculatorInput,
    bot: BotModel | None,
    model: type_model_name | None,
) -> str:
    """
    Calculator tool function for AgentTool.

    Args:
        input_data: Calculator input containing the expression
        bot: Bot model (not used for calculator)
        model: Model name (not used for calculator)

    Returns:
        str: Calculation result
    """
    return calculate_expression(input_data.expression)


# Backward compatibility alias
_calculate_expression = calculate_expression


# Create the calculator tool instance
calculator_tool = AgentTool(
    name="calculator",
    description="Perform mathematical calculations like addition, subtraction, multiplication, and division",
    args_schema=CalculatorInput,
    function=_calculator_function,
)
