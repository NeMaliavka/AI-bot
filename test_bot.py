import asyncio
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext
)
from dotenv import load_dotenv
import aiohttp
import json
import time
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s]: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Данные аутентификации
API_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHATBOT_ENDPOINT = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
SBERCLOUD_API_KEY = os.getenv('SBERCLOUD_API_KEY')

# Переменные для хранения токена и времени истечения срока действия
access_token = None
token_expiration_time = 0

# Хранилище контекста общения с каждым пользователем
USER_CONTEXT = {}

# Нормализация текста
def normalize_text(text):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text.lower(), language="russian")
    return " ".join(tokens)

# Очистка JSON от некорректных символов
def clean_json(data):
    cleaned_data = re.sub(r'\u0000|\ufeff|[^\x00-\xFF]+', '', data)
    return cleaned_data

# Загрузка обучающих данных
TRAINING_DATA = []  # Массив инструкций и ответов
INSTRUCTIONS = []  # Множество нормированных инструкций
ANSWERS = []  # Соответствующие ответы

try:
    with open('training_data.jsonl', encoding='utf-8') as file:
        lines = file.readlines()
        TRAINING_DATA = []
        for line in lines:
            cleaned_line = clean_json(line)
            try:
                entry = json.loads(cleaned_line)
                instruction = entry["instruction"]
                answer = entry["answer"]
                TRAINING_DATA.append({"instruction": instruction, "answer": answer})
                INSTRUCTIONS.append(normalize_text(instruction))
                ANSWERS.append(answer)
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка при чтении JSON: {e}")
        QUESTION_INDEX = {item['instruction'].lower(): item['answer'] for item in TRAINING_DATA}
except FileNotFoundError:
    logger.warning("Файл training_data.jsonl не найден. Функционал обучения отключается.")
    TRAINING_DATA = []
    QUESTION_INDEX = {}

# Векторизатор для нормализации текста
vectorizer = TfidfVectorizer(min_df=1, stop_words=None)  # Отключаем фильтрацию стоп-слов
INSTRUCTION_VECTORS = vectorizer.fit_transform(INSTRUCTIONS)

# Поиск лучшего ответа по обучению
async def get_training_answer(instruction):
    normalized_instruction = normalize_text(instruction)
    new_instruction_vector = vectorizer.transform([normalized_instruction])
    similarities = cosine_similarity(new_instruction_vector, INSTRUCTION_VECTORS)[0]
    best_match_idx = similarities.argmax()
    max_similarity = similarities.max()
    if max_similarity > 0.3:  # Порог сходства
        return ANSWERS[best_match_idx]
    else:
        return None

# Информация о курсе
try:
    with open('data_for_train_AI-bot.txt', 'r', encoding="utf8") as file_txt:
        COURSE_INFO = file_txt.read()
except FileNotFoundError:
    logger.error("Файл с информацией о курсе не найден.")
    COURSE_INFO = """
Мы предлагаем курс — «Основы программирования на Python». Он предназначен для детей от 10 до 17 лет и проводится в небольших группах (3–8 учеников). Курс охватывает основы языка Python и заканчивается разработкой собственного проекта. Продолжительность курса — от 20 до 30 занятий, в зависимости от потребностей ученика. Стоимость курса: от 13 000₽ до 19 500₽.
"""

async def get_access_token(session):
    global access_token, token_expiration_time
    try:
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': 'bdfd91f1-f0f7-4fbb-b3e2-1e60caefad5a',
            'Authorization': f'Bearer {SBERCLOUD_API_KEY}'
        }
        payload = {'scope': 'GIGACHAT_API_PERS'}
        
        async with session.post(API_URL, headers=headers, data=payload) as resp:
            if resp.status != 200:
                raise Exception(f"Ошибка при получении токена: {resp.status}, {await resp.text()}")
            
            response_data = await resp.json()
            access_token = response_data.get('access_token')
            expires_at_ms = response_data.get('expires_at')
            if not isinstance(expires_at_ms, int) or expires_at_ms <= 0:
                raise ValueError("Invalid expiration timestamp")
            token_expiration_time = expires_at_ms / 1000  # Переводим миллисекунды в секунды
            print("Access Token получен успешно.")
            return True
    except Exception as e:
        print(f"Ошибка при получении токена: {e}")
        return False

async def ask_gigachat(messages):
    global access_token, token_expiration_time
    current_time = time.time()
    if current_time >= token_expiration_time or access_token is None:
        print("Токен устарел или отсутствует. Обновление...")
        async with aiohttp.ClientSession() as session:
            success = await get_access_token(session)
            if not success:
                return None
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    payload = {
        "model": "gigachat",
        "messages": messages,
        "max_tokens": 1000
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(CHATBOT_ENDPOINT, headers=headers, json=payload) as resp:
            if resp.status != 200:
                logger.error(f"Ошибка при отправке сообщения: {resp.status}, {await resp.text()}")
                return None
            try:
                response_data = await resp.json()
                choices = response_data.get("choices", [])
                if choices:
                    answer = choices[0].get("message").get("content").strip()
                    return answer
                else:
                    return None
            except KeyError:
                logger.error("Ошибка в структуре JSON ответа.")
                return None

def filter_response(answer):
    if answer is None:
        return ""
    forbidden_keywords = ["Scratch", "JavaScript", "Kotlin"]  # Сюда включаем термины, которых быть не должно
    for keyword in forbidden_keywords:
        if keyword in answer:
            return ""
    return answer

async def generate_answer(user_id, user_message):
    # Сначала пытаемся найти точный ответ из обучающих данных
    answer_from_training = await get_training_answer(user_message)
    if answer_from_training:
        return answer_from_training
    
    # Если точного ответа нет, формируем запрос к GigaChat
    lower_case_query = user_message.lower()
    if lower_case_query in QUESTION_INDEX:
        answer = QUESTION_INDEX[lower_case_query]
    else:
        # Формирование запроса к GigaChat
        system_prompt = f"""\
Твоя задача — помогать родителям выбирать подходящие курсы программирования для детей и подростков.
ИНФОРМАЦИЯ О НАШИХ КУРСАХ:
{COURSE_INFO}
ВАЖНЫЕ ИНСТРУКЦИИ:
1. Ты МОЖЕШЬ использовать ТОЛЬКО данные, представленные выше. Никакая дополнительная информация не должна появляться в ответах.
2. Ответы должны быть КОНКРЕТНЫМИ и БЕЗ ЛИШНЕЙ ИНФОРМАЦИИ.
3. Никогда не допускай включения ЛЮБОЙ информации, которой НЕТ в описании курсов.
4. НИКАКИХ упоминаний иных курсов или услуг, которые не описаны в тексте выше.
5. В своих ответах, добавляй уточняющие вопросы для клиентов, чтобы продолжать диалог и помочь пользователю получить всю необходимую информацию о нашей школе и записаться на пробное занятие.
Формулируй ответы простым и доступным языком, подчеркивая преимущества существующих курсов. Задавай дополнительные вопросы в своих ответах, чтобы лучше контактировать с пользователем и быстрее вывести его на запись на пробное занятие.
"""
        history = USER_CONTEXT.get(user_id, [])  # Получаем предыдущую историю переписки
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        # Добавляем предыдущую историю после текущего запроса
        messages.extend(history)
        # Отправляем запрос
        raw_answer = await ask_gigachat(messages)
        # Фильтруем ответ
        filtered_answer = filter_response(raw_answer)
        # Если ответ валиден, продолжаем работу
        if filtered_answer:
            answer = filtered_answer
            # Обновляем историю переписки
            USER_CONTEXT[user_id] = messages + [{"role": "assistant", "content": answer}]
        else:
            answer = "Простите, не совсем понятно, о чём идёт речь."
    return answer

async def handle_message(update: Update, context: CallbackContext):
    message = update.message.text
    user_id = update.effective_user.id
    logger.debug(f"Пользователь {user_id} отправил сообщение: {message}")

    response = await generate_answer(user_id, message)
    # Проверяем, что ответ не пустой, прежде чем отправлять
    if response.strip():
        await update.message.reply_text(response)
        logger.info(f"Ответ пользователю {user_id}: {response}")
    else:
        logger.warning(f"Ответ пуст, сообщение не было отправлено пользователю {user_id}.")

# Создаем приложение и добавляем обработчики
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

# Запускаем бот
app.run_polling()