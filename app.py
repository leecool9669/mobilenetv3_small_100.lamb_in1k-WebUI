# -*- coding: utf-8 -*-
"""
MobileNetV3-Small-100 (LAMB/ImageNet-1k) 图像分类 WebUI 演示。
不加载真实模型权重，仅提供前端界面展示与交互流程。
"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载模型，实际不下载权重，仅用于界面演示。"""
    return "模型状态：mobilenetv3_small_100.lamb_in1k 已就绪（演示模式，未加载真实权重）"


def fake_classify(image):
    """模拟图像分类：不执行真实推理，仅返回示例 Top-5 结果说明。"""
    if image is None:
        return "请先上传一张图片，然后点击「执行分类（演示）」查看示例输出。"
    return (
        "[演示] 已对输入图像进行归一化与 224×224 缩放（未加载真实模型）。\n\n"
        "Top-5 分类结果示例（占位）：\n"
        "  1. 虎斑猫 (tabby cat) — 置信度 0.42\n"
        "  2. 埃及猫 (Egyptian cat) — 置信度 0.28\n"
        "  3. 斑猫 (tabby) — 置信度 0.12\n"
        "  4. 洗衣机 (washer) — 置信度 0.06\n"
        "  5. 纸巾盒 (tissue box) — 置信度 0.03\n\n"
        "加载真实 mobilenetv3_small_100.lamb_in1k 后，将在此显示 ImageNet-1k 的实际预测结果。"
    )


def build_ui():
    with gr.Blocks(title="MobileNetV3-Small-100 WebUI") as demo:
        gr.Markdown("## MobileNetV3-Small-100 (LAMB/ImageNet-1k) · 图像分类 WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示基于 LAMB 优化器在 ImageNet-1k 上训练的 MobileNetV3-Small 模型的典型使用流程，"
            "包括模型加载状态与图像分类结果可视化。输入图像将按 224×224 进行预处理并得到 Top-K 类别预测。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("图像分类"):
                gr.Markdown("上传一张图片，模型将输出 ImageNet-1k 类别下的 Top-5 预测（当前为演示占位结果）。")
                img_in = gr.Image(label="输入图像", type="pil")
                cls_out = gr.Textbox(label="分类结果说明", lines=12, interactive=False)
                run_btn = gr.Button("执行分类（演示）")
                run_btn.click(fn=fake_classify, inputs=img_in, outputs=cls_out)

        gr.Markdown("---\n*说明：当前为轻量级演示界面，未实际下载与加载 mobilenetv3_small_100.lamb_in1k 模型参数。*")

    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=8860, share=False)


if __name__ == "__main__":
    main()
