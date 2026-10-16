"""
Chainlit UI: Giao diện tương tác người dùng, hiển thị PDF và Báo cáo rủi ro hợp đồng
"""

import chainlit as cl
from loguru import logger

@cl.on_chat_start
async def start():
    logger.info("Initializing Chainlit user session...")
    await cl.Message(
        content="🛡️ **Chào mừng bạn đến với LexiGuard!**\n\nHệ thống Multi-Agent rà soát lỗ hổng hợp đồng và đối chiếu pháp lý (OCR + Hybrid GraphRAG).\n\nVui lòng tải lên file hợp đồng (.pdf, .docx) để bắt đầu phân tích rủi ro và nhận đề xuất redline tự động."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Received user message: {message.content}")
    # Placeholder interaction logic
    await cl.Message(
        content=f"Đang xử lý yêu cầu của bạn: {message.content}..."
    ).send()
