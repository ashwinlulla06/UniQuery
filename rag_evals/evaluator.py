import gradio as gr
import pandas as pd
import plotly.express as px
from collections import defaultdict
from dotenv import load_dotenv

from rag_evals.evaluating_retrieval import evaluate_all_retrieval

load_dotenv(override=True)

# Color coding threshold - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type in ["mrr", "recall", "hit", "precision"]:
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"
    return "black"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def run_retrieval_evaluation(progress=gr.Progress()):
    """Run retrieval evaluation and yield updates."""
    total_mrr = 0.0
    total_ndcg = 0.0
    total_recall = 0.0
    total_hit = 0.0
    total_precision = 0.0
    category_scores = defaultdict(lambda: defaultdict(list))
    count = 0

    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr_score
        total_ndcg += result.ndcg_score
        total_recall += result.recall_score
        total_hit += result.hit_score
        total_precision += result.precision_score
        curr_category = test.category

        category_scores[curr_category]["MRR"].append(result.mrr_score)
        category_scores[curr_category]["nDCG"].append(result.ndcg_score)
        category_scores[curr_category]["Recall@K"].append(result.recall_score)
        category_scores[curr_category]["Hit Rate@K"].append(result.hit_score)
        category_scores[curr_category]["Precision@1"].append(result.precision_score)

        # Update progress bar only
        progress(prog_value, desc=f"Evaluating test {count}...")

    if count == 0:
        empty_html = """
        <div style="padding: 20px; text-align: center; color: #999;">
            No answerable test questions were found.
        </div>
        """
        return empty_html, None

    # Calculate final averages
    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_recall = total_recall / count
    avg_hit = total_hit / count
    avg_precision = total_precision / count

    # Create final summary metrics HTML
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Recall@K", avg_recall, "recall")}
        {format_metric_html("Hit Rate@K", avg_hit, "hit")}
        {format_metric_html("Precision@1", avg_precision, "precision")}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    # Create final bar chart data
    category_data = []
    for category, metrics in category_scores.items():
        for metric, scores in metrics.items():
            category_data.append(
                {
                    "Category": category,
                    "Metric": metric,
                    "Average Score": sum(scores) / len(scores),
                }
            )

    df = pd.DataFrame(category_data)

    figure = px.bar(
        df,
        x="Category",
        y="Average Score",
        color="Metric",
        barmode="group",
        title="Average Retrieval Metrics by Category",
        range_y=[0, 1],
        template="plotly_dark",
    )

    figure.update_layout(
        height=450,
        xaxis_tickangle=-20,
        legend_title_text="Metric",
    )

    return final_html, figure


def main():
    """Launch the Gradio evaluation app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for the UniQuery RAG system")

        # RETRIEVAL SECTION
        gr.Markdown("## 🔍 Retrieval Evaluation")

        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )

            with gr.Column(scale=1):
                retrieval_chart = gr.Plot()

        # # ANSWERING SECTION
        # gr.Markdown("## 💬 Answer Evaluation")

        # answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        # with gr.Row():
        #     with gr.Column(scale=1):
        #         answer_metrics = gr.HTML(
        #             "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
        #         )

        #     with gr.Column(scale=1):
        #         answer_chart = gr.BarPlot(
        #             x="Category",
        #             y="Average Accuracy",
        #             title="Average Accuracy by Category",
        #             y_lim=[1, 5],
        #             height=400,
        #         )

        # Wire up the evaluations
        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart],
        )

        # answer_button.click(
        #     fn=run_answer_evaluation,
        #     outputs=[answer_metrics, answer_chart],
        # )

    app.launch(inbrowser=True, share=True)


if __name__ == "__main__":
    main()
