# Документирование проекта

Проект представляет собой телеграм-бота, интегрирующего внешние ресурсы (например, GigaChat) для эффективного консультирования пользователей по выбору курсов программирования для детей и подростков.

## Основные цели проекта:
- Интерактивное общение с родителями, выбирающими курсы для своих детей.
- Поддержка обмена знаниями и помощь в принятии решений.
- Эффективная интеграция внешних сервисов и предоставление качественных консультационных услуг.

## Архитектура и основные этапы:

1. **Загрузка и настройка**:
   - Загрузка токенов и конфиденциальных данных из файла `.env` с помощью модуля `dotenv`.
   - Настройка механизма ведения журналов (логирования) с уровнем `INFO`.
   - Подготовка интерфейса Telegram Bot API с использованием токена, полученного из среды.

2. **Работа с данными**:
   - Чтение обучающих данных из файла `training_data.jsonl`, содержащего пары вопросов и ответов.
   - Предварительная обработка данных (нормализация текста, очистка от посторонних символов).

3. **Интеграция с API**:
   - Использование внешних ресурсов (GigaChat) для генерации ответов на запросы пользователей.
   - Авторизация и управление временем жизни токена для безопасного доступа к API.

4. **Коммуникация с пользователями**:
   - Прием сообщений от пользователей через Telegram.
   - Генерирование и отправка соответствующих ответов на основе обучающей базы данных или путем обращения к GigaChat.

5. **Мониторинг и обработка ошибок**:
   - Ведется журнал операций для отладки и мониторинга производительности.
   - Обеспечивается надежный механизм восстановления после различных видов ошибок.

---

## Основные функции и их предназначение:

### 1. `normalize_text(text)`

- **Цель**: Преобразование входящего текста в унифицированную форму (преобразование регистра, токенизация).
- **Метод**: Используется библиотека NLTK для токенизации текста на русский язык.

```python
def normalize_text(text):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text.lower(), language="russian")
    return " ".join(tokens)
```

---

### 2. `clean_json(data)`

- **Цель**: Очистка JSON-данных от некорректных символов и неподдерживаемых символов Юникода.
- **Метод**: Применение регулярного выражения для удаления нежелательных символов.

```python
def clean_json(data):
    cleaned_data = re.sub(r'\u0000|\ufeff|[^\x00-\xFF]+', '', data)
    return cleaned_data
```

---

### 3. `get_training_answer(instruction)`

- **Цель**: Нахождение наилучшего ответа на запрос пользователя среди обучающих данных.
- **Алгоритм**: Использование векторизации TF-IDF и алгоритмов нахождения ближайшего соседа (метрика близости).

```python
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
```

---

### 4. `get_access_token(session)`

- **Цель**: Получение свежего токена авторизации для последующего использования API GigaChat.
- **Механизм**: Отправка запроса к серверу OAuth и последующее сохранение токена и времени его завершения.

```python
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
```

---

### 5. `ask_gigachat(messages)`

- **Цель**: Отправка запроса на сервер GigaChat и получение соответствующего ответа.
- **Механизм**: Осуществление асинхронного HTTP-запроса к API и последующая обработка ответа.

```python
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
```

---

### 6. `filter_response(answer)`

- **Цель**: Фильтрация ответа, удаляя нежелательные ключевые слова.
- **Метод**: Проверка присутствия определенных слов и возвращение пустой строки, если такие обнаружены.

```python
def filter_response(answer):
    if answer is None:
        return ""
    forbidden_keywords = ["Scratch", "JavaScript", "Kotlin"]  # Сюда включаем термины, которых быть не должно
    for keyword in forbidden_keywords:
        if keyword in answer:
            return ""
    return answer
```

---

### 7. `generate_answer(user_id, user_message)`

- **Цель**: Генерация ответа для конкретного пользователя.
- **Алгоритм**: Проверка наличия готовых ответов среди обучающих данных, обращение к GigaChat для генерации нового ответа, если прямого ответа не найдено.

```python
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
        system_prompt = f"... {COURSE_INFO}"
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
```

---

### 8. `handle_message(update, context)`

- **Цель**: Главный обработчик сообщений, приходящих от пользователей.
- **Механизм**: Прием сообщения от пользователя, передача его в генератор ответов и отправка результата пользователю.

```python
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
```

