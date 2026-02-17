"""
Gradio UI for the Business Intelligence Agent Pipeline.

This app demonstrates Google ADK's SequentialAgent pattern:
1. Text-to-SQL Agent (standalone)
2. SQL execution via BIService
3. Insight Pipeline (SequentialAgent: Visualization → Explanation)
"""

import gradio as gr
import asyncio
import os
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from google.genai import types

# Import root agent runner
from bi_agent import root_runner

# Load environment variables from bi_agent/.env
load_dotenv(dotenv_path='bi_agent/.env')


async def run_bi_pipeline_async(user_question: str):
    """
    Run the complete BI pipeline using root_runner.

    This function executes the entire BI pipeline:
    1. Text-to-SQL: Generate SQL from question
    2. SQL Execution: Execute query against database
    3. Data Formatting: Prepare results
    4. Visualization: Generate Altair chart
    5. Explanation: Provide plain-language insights

    Args:
        user_question: Natural language question from the user

    Returns:
        Dictionary with keys: sql_query, query_results, chart_spec, explanation_text
    """
    # Create session
    session = await root_runner.session_service.create_session(
        user_id='user',
        app_name='bi_agent'
    )

    # Create user message
    content = types.Content(
        role='user',
        parts=[types.Part(text=user_question)]
    )

    # Run the complete pipeline
    events_async = root_runner.run_async(
        user_id='user',
        session_id=session.id,
        new_message=content
    )

    # Extract results from state
    results = {}
    async for event in events_async:
        if event.actions and event.actions.state_delta:
            for key, value in event.actions.state_delta.items():
                results[key] = value

    return results


async def process_request_async(message: str):
    """
    Process user request through the BI pipeline using root_runner.

    The root_agent handles the complete pipeline:
    1. Text-to-SQL Agent → Generates SQL from question
    2. SQL Executor Agent → Executes SQL against database
    3. Data Formatter Agent → Formats results
    4. Insight Pipeline → Visualization + Explanation

    Args:
        message: User's natural language question

    Returns:
        Tuple of (sql_query, df, chart, explanation_text)
    """
    try:
        # Validate input
        if not message.strip():
            return "Error: Please enter a question", None, None, "Error: No question provided"

        # Run the complete BI pipeline
        results = await run_bi_pipeline_async(message)

        # Extract SQL query
        sql_query = results.get('sql_query', '')

        # Clean up SQL query (remove markdown if present)
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()

        # Extract query results
        query_results_str = results.get('query_results', '{}')
        try:
            import json
            query_results = json.loads(query_results_str) if isinstance(query_results_str, str) else query_results_str
        except:
            query_results = {'success': False, 'data': [], 'error': 'Failed to parse query results'}

        # Check if query execution was successful
        if not query_results.get('success', False):
            error_msg = query_results.get('error', 'Unknown error')
            sql_query = f"-- Error executing query\n{sql_query}\n\n-- Error: {error_msg}"
            return sql_query, None, None, f"Error executing query: {error_msg}"

        # Convert query results to DataFrame
        data_list = query_results.get('data', [])
        if not data_list:
            df = pd.DataFrame()
            return sql_query, df, None, "The query executed successfully but returned no data."

        df = pd.DataFrame(data_list)

        # Extract chart specification and explanation
        chart_spec = results.get('chart_spec', '')
        explanation_text = results.get('explanation_text', '')

        # Execute chart specification
        chart = None
        if chart_spec:
            try:
                chart_spec_clean = chart_spec.strip()
                if chart_spec_clean.startswith("```python"):
                    chart_spec_clean = chart_spec_clean.replace("```python", "").replace("```", "").strip()
                elif chart_spec_clean.startswith("```"):
                    chart_spec_clean = chart_spec_clean.replace("```", "").strip()

                # Create namespace and execute chart code
                namespace = {
                    'alt': alt,
                    'pd': pd,
                    'df': df,
                    'data': df.to_dict(orient='records')
                }

                exec(chart_spec_clean, namespace)

                if 'chart' in namespace:
                    chart = namespace['chart']
            except Exception as e:
                print(f"Chart generation error: {str(e)}")
                import traceback
                traceback.print_exc()

        # Return all four outputs
        return sql_query, df, chart, explanation_text

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Full error: {e}")
        import traceback
        traceback.print_exc()
        return error_msg, None, None, error_msg


def process_request(message: str):
    """
    Synchronous wrapper for Gradio.

    Database credentials are read from environment variables in bi_agent/.env
    """
    try:
        sql_query, df, chart, explanation = asyncio.run(
            process_request_async(message)
        )
        return sql_query, df, chart, explanation
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, None, None, error_msg


# ============================================================================
# Gradio UI
# ============================================================================

with gr.Blocks(title="Business Intelligence Agent") as demo:
    gr.Markdown("""
    # Business Intelligence Agent (Google ADK)

    This demo uses **Google ADK's root_agent SequentialAgent**:

    1. **Text-to-SQL Agent** → Generates SQL from natural language
    2. **SQL Executor Agent** → Executes SQL against database
    3. **Data Formatter Agent** → Prepares results for visualization
    4. **Insight Pipeline** (**SequentialAgent**) → Visualization Agent → Explanation Agent

    Database credentials are configured in `bi_agent/.env`

    Enter your question below and click "Analyze Data".
    """)

    with gr.Row():
        user_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., 'What are the top 10 products by price?'",
            lines=3
        )

    with gr.Row():
        submit_btn = gr.Button("Analyze Data", variant="primary")
        clear_btn = gr.Button("Clear")

    gr.Markdown("## Results")

    # Four output panels
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Generated SQL")
            sql_output = gr.Code(
                label="SQL Query",
                language="sql",
                value="-- Waiting for input..."
            )

        with gr.Column(scale=1):
            gr.Markdown("### Query Results")
            data_output = gr.DataFrame(
                label="Data Table",
                wrap=True
            )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Visualization")
            chart_output = gr.Plot(label="Chart")

        with gr.Column(scale=1):
            gr.Markdown("### Insights")
            explanation_output = gr.Markdown(
                value="*Waiting for input...*"
            )

    # Examples
    gr.Examples(
        examples=[
            ["What are the top 10 products by transfer price?"],
            ["Show me the product categories and their average prices"],
            ["List all products in the Bikes category"],
            ["How many products are there in each category?"],
            ["What is the most expensive product?"],
        ],
        inputs=user_input
    )

    # Button actions
    submit_btn.click(
        fn=process_request,
        inputs=[user_input],
        outputs=[sql_output, data_output, chart_output, explanation_output]
    )

    clear_btn.click(
        fn=lambda: (
            "",
            "-- Waiting for input...",
            None,
            None,
            "*Waiting for input...*"
        ),
        inputs=None,
        outputs=[user_input, sql_output, data_output, chart_output, explanation_output]
    )


if __name__ == "__main__":
    demo.launch()
