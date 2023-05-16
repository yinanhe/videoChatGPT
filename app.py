import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from conversation import Chat

# videochat
from utils.config import Config
from utils.easydict import EasyDict
from models.videochat import VideoChat


# ========================================
#             Model Initialization
# ========================================
def init_model():
    os.system('wget -P /home/xlab-app-center/model/ https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth')
    os.system('wget -P /home/xlab-app-center/model/ https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth')
    os.system('wget -P /home/xlab-app-center/model/ https:///models/yinanhe/videochat_13b/weight//videochat_13b -O videochat.pth')
    print('Initializing VideoChat')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()
    chat = Chat(model)
    print('Initialization Finished')
    return chat


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_img(gr_img, gr_video, chat_state, num_segments):
    # print(gr_img, gr_video)
    chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    img_list = []
    if gr_img is None and gr_video is None:
        return None, None, gr.update(interactive=True), chat_state, None
    if gr_video: 
        llm_message, img_list, chat_state = chat.upload_video(gr_video, chat_state, img_list, num_segments)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list
    if gr_img:
        llm_message, img_list,chat_state = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    #print(chat_state)
    chat_state =  chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(gr_img, gr_video,chatbot, chat_state, img_list, num_beams, temperature):
    llm_message,llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    chatbot[-1][1] = llm_message
    print(f"========{gr_img}##<BOS>##{gr_video}========")
    print(chat_state,flush=True)
    print(f"========{gr_img}##<END>##{gr_video}========")
    # print(f"Answer: {llm_message}")
    return chatbot, chat_state, img_list


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

title = """<h1 align="center"><a href="https://github.com/OpenGVLab/Ask-Anything"><img src="https://i.328888.xyz/2023/05/11/iqrAkZ.md.png" alt="Ask-Anything" border="0" style="margin: 0 auto; height: 100px;" /></a> </h1>"""
description ="""
        <p> VideoChat, an end-to-end chat-centric video understanding system powered by <a href='https://github.com/OpenGVLab/InternVideo'>InternVideo</a>. It integrates video foundation models and large language models via a learnable neural interface, excelling in spatiotemporal reasoning, event localization, and causal relationship inference.</p>
        <div style='display:flex; gap: 0.25rem; '>
        <a src="https://img.shields.io/badge/Github-Code-blue?logo=github" href="https://github.com/OpenGVLab/Ask-Anything"> <img src="https://img.shields.io/badge/Github-Code-blue?logo=github">
        <a src="https://img.shields.io/badge/cs.CV-2305.06355-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2305.06355"> <img src="https://img.shields.io/badge/cs.CV-2305.06355-b31b1b?logo=arxiv&logoColor=red">
        <a src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat" href="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/papers/media/wechat_group.jpg"> <img src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat">
        <a src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord" href="https://discord.gg/A2Ex6Pph6A"> <img src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord"> </div>
        """


with gr.Blocks(title="InternVideo-VideoChat!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image") as img_part:
                with gr.Tab("Video", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload")#.style(height=320)
                with gr.Tab("Image", elem_id='image_tab'):
                    up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload")#.style(height=320)
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            
            num_segments = gr.Slider(
                minimum=8,
                maximum=64,
                value=8,
                step=1,
                interactive=True,
                label="Video Segments",
            )
        
        
        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State(EasyDict({
                "system": "",
                "roles": ("Human", "Assistant"),
                "messages": [],
                "sep": "###"
            }))
            img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='VideoChat')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False).style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear️")     
    
    chat = init_model()
    upload_button.click(upload_img, [up_image, up_video, chat_state, num_segments], [up_image, up_video, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [up_image, up_video, chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [up_image, up_video,chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, up_image, up_video, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(server_name="0.0.0.0", favicon_path='bot_avatar.jpg', enable_queue=True)
