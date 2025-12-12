from quart import Quart, request, jsonify, send_file, session
from quart_cors import cors
import os
import re
import asyncio
from asyncio import TimeoutError as AsyncTimeoutError
import logging
import csv
import json
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import List, Dict, Any, Tuple, Optional
from langchain_gigachat.chat_models import GigaChat
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import uuid
import sys

# Определяем базовую директорию для Netlify
if 'LAMBDA_TASK_ROOT' in os.environ:
    # Мы в Netlify Functions
    BASE_DIR = os.environ.get('LAMBDA_TASK_ROOT', '')
else:
    # Локальная разработка
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#   Эти убрать потом, вырезать
from random_shift_parser import main_jf_advanced
from pdf_converter import create_resume_pdf

app = Quart(__name__)
app = cors(app, allow_origin="*")
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configuration
CHECK_EVERY = 10
semaphore = asyncio.Semaphore(5)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Available user IDs
AVAILABLE_USER_IDS = [
    "10022136282",
    "67651241646", 
    "14668188501",
    "27577193111",
    "Без аккаунта"
]

# Financial resources
FINANCIAL_RESOURCES = {
    "sbersova": "https://sbersova.ru/ (сайт сбербанка про финансовому обучению)",
    "fincult": "https://fincult.info/ (общий сайт по фин. обучению)",
    "nalog": "https://www.nalog.gov.ru/rn77/taxation/taxes/ndfl/nalog_vichet/ (сайт рассказывающий о налоговых вычетах)"
}
# LQDx3K6ssVBFzB8Ywg9VNr10iHj3tanT
# Initialize GigaChat
giga = GigaChat(
    credentials=os.environ.get('GIGACHAT_CREDENTIALS', "Здесь ключ вставить"),
    model="GigaChat-2",
    verify_ssl_certs=False,
)

mistral = ChatMistralAI(
    api_key=os.environ.get('MISTRAL_API_KEY', "your-mistral-api-key-here"),
    model="mistral-large-latest",  # Или другая модель, например "codestral-latest"
    temperature=0.7,
    max_tokens=1000,
)

# Store user sessions in memory
user_sessions = {}
user_notifications = {}

# УДАЛЕНО: active_connections = {}

def limit_messages_history(messages: List, max_human: int = 7, max_ai: int = 7) -> List:
    """Оптимизированная версия ограничения истории сообщений"""
    if not messages:
        return []
    
    system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
    other_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    
    # Сохраняем последние сообщения каждого типа
    human_msgs = [msg for msg in other_msgs if isinstance(msg, HumanMessage)][-max_human:]
    ai_msgs = [msg for msg in other_msgs if isinstance(msg, AIMessage)][-max_ai:]
    
    # Объединяем и сортируем по порядку
    all_recent = sorted(human_msgs + ai_msgs, 
                       key=lambda x: other_msgs.index(x) if x in other_msgs else 0)
    
    return system_msgs + all_recent

# Debug function to check session state3
def debug_session(session_id, operation):
    print(f"Session {session_id}: {operation}")
    if session_id in user_sessions:
        print(f"  - User ID: {user_sessions[session_id].get('selected_uid', 'None')}")
        print(f"  - Messages count: {len(user_sessions[session_id].get('messages', []))}")

def cleanup_old_sessions():
    """Очистка старых сессий"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session_data in user_sessions.items():
        if (current_time - session_data['last_activity']).total_seconds() > 3600:  # 1 час
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del user_sessions[session_id]
        logger.info(f"Удалена expired сессия: {session_id}")

async def get_response(complete_data, user_input: str, messages: List, user_id: int) -> Tuple[str, int]:
    """Оптимизированная версия получения ответа от GigaChat"""
    try:
        human_message = HumanMessage(content=user_input)
        current_messages = messages + [human_message]
        
        async with semaphore:
            response = await asyncio.wait_for(
                giga.ainvoke(current_messages),
                timeout=30.0
            )
        
        # Обновляем историю сообщений
        ai_message = AIMessage(content=response.content)
        messages.extend([human_message, ai_message])
        
        # Ограничиваем историю
        limited_messages = limit_messages_history(messages)
        messages.clear()
        messages.extend(limited_messages)
        
        token_usage = getattr(response, 'response_metadata', {}).get('token_usage', {})
        return response.content, token_usage.get('total_tokens', 0)
    
    except asyncio.TimeoutError:
        return "Извините, запрос занял слишком много времени. Пожалуйста, попробуйте позже.", 0
    except Exception as e:
        print(f"Error in get_response: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса", 0
        
async def is_valid_profession(profession: str) -> bool:
    """Проверяет, является ли текст валидной профессией"""
    try:
        prompt = f"Является ли '{profession}' названием реальной профессии или специальности? Ответь только 'да' или 'нет'."
        
        messages = [HumanMessage(content=prompt)]
        async with semaphore:
            response = await giga.ainvoke(messages)
        
        answer = response.content.strip().lower()
        return 'да' in answer
    except Exception as e:
        print(f"Error in profession validation: {e}")
        return True  # В случае ошибки принимаем как валидную

async def extract_job_info(text: str, human_data: str, selected_uid) -> str:
    if selected_uid != "Без аккаунта":
        prompt = f"""
Из следующего текста определи профессию и город для поиска работы. Верни ответ поисковым запросом по поиску работы.
БОЛЬШЕ НИЧЕГО НЕ ВЫВОДИ, ЕСЛИ ПОЛЬЗОВАТЕЛЬ ЛИЧНО УКАЗЫВАЕТ ДРУГИЕ ДАННЫЕ ТО ОНИ В ПРИОРИТЕТЕ, не добавляй никаких знаков - выделителей
Вот пример ответа: механик в хабаровске вакансии
Если считаешь, что в тексте какая то информация не дана используй эту сводку, там есть и специальность и город: {human_data}
Текст: {text}
"""
    else:
        prompt = f"""Из следующего текста определи профессию и город для поиска работы. Верни ответ поисковым запросом по поиску работы.
        БОЛЬШЕ НИЧЕГО НЕ ВЫВОДИ, ЕСЛИ ПОЛЬЗОВАТЕЛЬ ЛИЧНО УКАЗЫВАЕТ ДРУГИЕ ДАННЫЕ ТО ОНИ В ПРИОРИТЕТЕ, не добавляй никаких знаков - выделителей
Вот пример ответа: механик в хабаровске вакансииТекст: {text}. Если в тексте нет данных верни любые"""
    
    messages = [HumanMessage(content=prompt)]
    async with semaphore:
        response = await giga.ainvoke(messages)
    return response.content

async def prepare_interview(vacancy: str, experience: str) -> str:
    prompt = f"""
Ты профессиональный HR-менеджер.
Подготовь список из 5-7 теоретических вопросов для собеседования на должность {vacancy} для кандидата с опытом: {experience}.
Так же задай несколько вопросов на soft навыки, пусть не всё будет <хардами>
"""
    
    messages = [HumanMessage(content=prompt)]
    try:
        async with semaphore:
            response = await giga.ainvoke(messages)
        return response.content
    except Exception as e:
        return f"К сожалению, не могу подготовить вопросы для собеседования сейчас. Ошибка: {str(e)}"

async def create_resume(profession: str, experience: str, additional: str, user_name: str = None, city: str = None) -> str:
    # Формируем промпт с учетом данных пользователя
    contact_info = ""
    if user_name:
        contact_info += f"- Имя: {user_name}\n"
    if city:
        contact_info += f"- Город: {city}\n"
    
    prompt = f"""
Составь подробное резюме для профессии {profession}.

Информация от пользователя:
{contact_info if contact_info else ""}- Опыт работы: {experience}
- Дополнительная информация: {additional}

Включи следующие разделы:
1. Контактная информация{'' if not contact_info else ' (уже включает имя и город из данных пользователя)'}
2. Цель поиска работы  
3. Опыт работы
4. Образование
5. Ключевые навыки и компетенции
6. Дополнительная информация (если есть)

{'Учти, что в контактной информации уже есть имя и город пользователя, не дублируй их.' if contact_info else ''}
"""
    
    messages = [HumanMessage(content=prompt)]
    try:
        async with semaphore:
            response = await giga.ainvoke(messages)
            print("Создано резюме:", response.content[:100] + "...")
            
            # Создаем PDF, но также возвращаем текстовое содержимое
            try:
                # create_resume_pdf(response.content, f"resume_{user_name}.pdf")
                print("PDF резюме создан успешно")
            except Exception as e:
                print(f"Ошибка при создании PDF: {e}")
                # Продолжаем работу даже если PDF не создался
            
            return response.content  # ВАЖНО: возвращаем текст резюме
            
    except Exception as e:
        print(f"Error in create_resume: {e}")
        return "К сожалению, не могу помочь с составлением резюме сейчас."

async def prepare_fin_analysis(results: list, session_data: dict) -> str:
    now = datetime.now()
    """Подготовка финансового анализа с логированием"""
    print(f"Starting financial analysis with {len(results)} results")
    
    # Рассчитываем общую сумму платежей
    total_payments = calculate_total_payments(results)
    upcoming_payments = get_upcoming_payments(results)
    
    prompt = f"""Твоя задача выдать пользователю анализ его ситуации по выданным данным. Ты сообщаешь результаты клиенту лично, будь вежлив, ОБРАЩАЙСЯ К КЛИЕНТУ НА ВЫ. Учти, что клиент УЖЕ прошел реструктуризацию, твоя задача проанализировать его ситуацию и выдать список рекомендаций по улучшению ситуации, составить примерный план.
ВСЕ ДЕНЕЖНЫЕ ЗНАЧЕНИЯ В РУБЛЯХ
Важная информация для анализа:
- Общая сумма предстоящих платежей: {format_amount(total_payments)} руб.
- Ближайшие платежи: {upcoming_payments}
Можешь просто вывести информацию о платежах, не сравнивай достаточно ли средств.

СЕГОДНЯ {now.strftime('%d.%m.%Y')}
Не создавай таблицы и не оперируй точными датами, обходись примерными сроками, вместо таблиц - делай списки. Если у клиента несколько обязательств - распиши каждую из них. Так же слева есть раздел с функциями по поиску работы, подготовке к собеседованию и составлением резюме. Постарайся не использовать сложные термины или формулировки.
Вот данные: {str(results)}.
Попробуй обойтись 500 символов.
"""
    
    print(f"Financial analysis prompt: {prompt[:200]}...")
    
    messages = [HumanMessage(content=prompt)]
    try:
        async with semaphore:
            response = await giga.ainvoke(messages)
            print(f"GigaChat response received: {len(response.content)} characters")
            
            human_msg = HumanMessage(content="Мой финансовый анализ")
            ai_msg = AIMessage(content=response.content)
            session_data['messages'].extend([human_msg, ai_msg])
            
            # Ограничиваем историю сообщений
            limited_messages = limit_messages_history(session_data['messages'])
            session_data['messages'].clear()
            session_data['messages'].extend(limited_messages)
            
        return response.content
    except Exception as e:
        print(f"Error in prepare_fin_analysis: {e}")
        return "К сожалению, не могу выполнить анализ сейчас."    

def get_csv_path():
    """Определяет путь к CSV файлу в зависимости от среды выполнения"""
    possible_paths = [
        os.path.join(BASE_DIR, 'show_case_database.csv'),
        os.path.join(BASE_DIR, 'functions', 'show_case_database.csv'),
        os.path.join(BASE_DIR, 'netlify', 'functions', 'show_case_database.csv'),
        'show_case_database.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"CSV файл найден: {path}")
            return path
    
    print("CSV файл не найден ни по одному из путей:")
    for path in possible_paths:
        print(f"  - {path}")
    return 'show_case_database.csv'  # fallback

def parse_csv_table(file_path: str, search_value: str, delimiter: str = ';', encoding: str = 'cp1251') -> Tuple[List[Dict], List[Dict]]:
    """
    Парсит CSV таблицу и возвращает ВСЕ обязательства пользователя
    """
    all_obligations = []
    actual_file_path = get_csv_path()
    
    print(f"Пытаемся прочитать CSV файл: {actual_file_path}")
    
    try:
        with open(actual_file_path, 'r', encoding=encoding, newline='') as file:
            reader = csv.reader(file, delimiter=delimiter)
            headers = next(reader, None)

            for row_num, row in enumerate(reader, 1):
                if not row or row[0] != search_value:
                    continue

                # Создаем запись обязательства с учетом новой структуры данных
                obligation = {
                    'row_num': row_num,
                    'user_name': row[1] if len(row) > 1 else '',
                    'contract': row[2] if len(row) > 2 else '',
                    'credit_product': row[3] if len(row) > 3 else '',
                    'restruct_date': row[4] if len(row) > 4 else '',
                    'rest_reason': row[5] if len(row) > 5 else '',
                    'payment_day': row[6] if len(row) > 6 else '',
                    'payment_amount': row[7] if len(row) > 7 else '',
                    'account_balance': row[8] if len(row) > 8 else '',
                    'job_info': row[9] if len(row) > 9 else '',
                    'income_before': row[10] if len(row) > 10 else '',
                    'income_after': row[11] if len(row) > 11 else '',
                    'social_payments': row[12] if len(row) > 12 else '',
                    'address': row[13] if len(row) > 13 else '',
                    'other_data': row[14] if len(row) > 14 else ''
                }
                
                if headers:
                    obligation.update({headers[i]: row[i] for i in range(min(len(headers), len(row)))})
                
                all_obligations.append(obligation)

        print(f"Найдено обязательств для пользователя {search_value}: {len(all_obligations)}")
        return all_obligations, all_obligations[-1] if all_obligations else {}

    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return [], {}

def get_user_session(session_id, user_id=None):
    """Get or create user session"""
    if session_id not in user_sessions:
        if user_id:
            return initialize_user_session(session_id, user_id)
        else:
            return initialize_user_session(session_id, "Без аккаунта")
    
    user_sessions[session_id]['last_activity'] = datetime.now()
    debug_session(session_id, "retrieved")
    return user_sessions[session_id]

async def handle_message(user_input: str, session_data: dict) -> Tuple[str, List[Dict]]:
    """Handle regular chat messages with buttons ONLY for completed responses"""
    print(f"Handling message: {user_input}")
    
    # Проверяем, есть ли активный процесс сбора данных
    if session_data.get('current_process'):
        response_text = await handle_process_step(user_input, session_data, session_data['selected_uid'])
        # Во время активного процесса не показываем кнопки
        return response_text, []
    
    # Проверяем специальные запросы
    user_input_lower = user_input.lower()
    
    if user_input_lower in ['платежи', 'мои платежи', 'инфо о платежах']:
        response_text = await handle_payment_info_request(session_data)
    elif user_input_lower in ['баланс', 'мой баланс', 'средства']:
        response_text = await handle_balance_request(session_data)
    # Проверяем, не спрашивает ли пользователь о платежах
    elif any(keyword in user_input_lower for keyword in ['платеж', 'платить', 'сколько', 'сумма', 'деньги', 'оплата', 'задолженность']):
        response_text = await handle_payment_question(user_input, session_data)
    else:
        # Обычная обработка сообщения
        try:
            # Просто получаем ответ от модели без определения намерений
            reply, tokens = await get_response(
                session_data['complete_data_str'], 
                user_input, 
                session_data['messages'], 
                0
            )

            # Обновляем историю сообщений
            session_data['hist_r'] += f"human: {user_input}; ai: {reply}; "
            
            # Добавляем сообщения в историю и ограничиваем
            human_msg = HumanMessage(content=user_input)
            ai_msg = AIMessage(content=reply)
            session_data['messages'].extend([human_msg, ai_msg])
            
            # Ограничиваем историю сообщений
            limited_messages = limit_messages_history(session_data['messages'])
            session_data['messages'].clear()
            session_data['messages'].extend(limited_messages)
            
            response_text = reply
            
        except Exception as e:
            print(f"Error in handle_message: {e}")
            response_text = "Извините, произошла ошибка при обработке вашего сообщения."
    
    # Получаем кнопки для ответа ТОЛЬКО если нет активного процесса
    buttons = get_buttons_for_response(session_data, response_text, 'message')
    
    return response_text, buttons

async def handle_payment_question(user_input: str, session_data: dict) -> str:
    """Обрабатывает вопросы о платежах"""
    if session_data['selected_uid'] == "Без аккаунта":
        return "Для получения информации о платежах необходимо выбрать пользователя из списка."
    
    obligations = session_data.get('obligations', [])
    if not obligations:
        return "У вас нет данных о платежах в системе."
    
    total_payments = session_data.get('total_payments', 0)
    upcoming_payments = session_data.get('upcoming_payments', '')
    
    # Анализируем вопрос пользователя
    user_input_lower = user_input.lower()
    
    if 'сколько' in user_input_lower and 'платить' in user_input_lower:
        response = f"Общая сумма предстоящих платежей: {format_amount(total_payments)} руб.\n\n"
        
        if upcoming_payments:
            response += f"Ближайшие платежи:\n{upcoming_payments}\n\n"
        
        # Добавляем детализацию по каждому обязательству
        response += "Детализация по договорам:\n"
        for i, oblig in enumerate(obligations, 1):
            contract = oblig.get('contract', 'Не указан')
            payment_date = oblig.get('payment_day', 'Не указана')
            payment_amount = oblig.get('payment_amount', '0')
            amount_num = parse_amount(payment_amount)
            
            response += f"{i}. Договор {contract}: {format_amount(amount_num)} руб. (срок: {payment_date})\n"
        
        return response
    
    elif 'когда' in user_input_lower and 'платить' in user_input_lower:
        if upcoming_payments:
            return f"Ваши ближайшие платежи:\n{upcoming_payments}"
        else:
            return "У вас нет предстоящих платежей в ближайшее время."
    
    else:
        # Общая информация о платежах
        response = f"По вашим обязательствам:\n"
        response += f"- Общая сумма к оплате: {format_amount(total_payments)} руб.\n"
        response += f"- Ближайшие платежи: {upcoming_payments}\n\n"
        response += "Для получения подробной информации спросите 'Сколько мне нужно заплатить?' или 'Когда мне платить?'"
        
        return response

async def handle_process_step(user_input: str, session_data: dict, selected_uid: str) -> str:
    """Обрабатывает шаг активного процесса (резюме, поиск работы, собеседование) БЕЗ КНОПОК"""
    process = session_data['current_process']
    
    print(f"DEBUG handle_process_step: process={process}, step={session_data['current_step']}, user_input='{user_input}'")
    
    try:
        if process == 'resume':
            result = await handle_resume_process(user_input, session_data, session_data['current_step'], selected_uid)
        elif process == 'job_search':
            result = await handle_job_search_process(user_input, session_data, session_data['current_step'], selected_uid)
        elif process == 'interview':
            result = await handle_interview_process(user_input, session_data, session_data['current_step'], selected_uid)
        else:
            return "Неизвестный процесс"
        
        # ВАЖНО: Возвращаем только текст, без кнопок
        return str(result) if result is not None else "Произошла ошибка в процессе"
            
    except Exception as e:
        print(f"Error in handle_process_step: {e}")
        return "Извините, произошла ошибка. Пожалуйста, попробуйте еще раз."

async def handle_resume_process(user_input: str, session_data: dict, step: int, selected_uid: str) -> str:
    """Обрабатывает процесс создания резюме БЕЗ КНОПОК"""
    print(f"DEBUG: handle_resume_process - step: {step}, user_input: '{user_input}'")
    
    try:
        # Для зарегистрированных пользователей проверяем и подтверждаем данные из БД
        if selected_uid != "Без аккаунта" and not session_data.get('confirmed_data', {}).get('resume_confirmed'):
            return await confirm_db_data_for_resume(user_input, session_data)

        if step == 1:  # Шаг 1: Специальность
            if len(user_input.strip()) < 2:
                return "Пожалуйста, укажите вашу специальность более подробно (например: 'программист', 'бухгалтер', 'менеджер'):"
            
            session_data['process_data']['profession'] = user_input
            session_data['current_step'] = 2
            return "Отлично! Теперь опишите ваш опыт работы (продолжительность, проекты, обязанности):"
        
        elif step == 2:  # Шаг 2: Опыт работы
            session_data['process_data']['experience'] = user_input
            session_data['current_step'] = 3
            return "Хорошо! Теперь укажите любую дополнительную информацию (образование, сертификаты, навыки, достижения):"
        
        elif step == 3:  # Шаг 3: Дополнительная информация
            session_data['process_data']['additional'] = user_input
            
            # Создаем резюме с данными пользователя
            profession = session_data['process_data']['profession']
            experience = session_data['process_data']['experience']
            additional = session_data['process_data']['additional']
            
            # Получаем данные пользователя из сессии
            user_name = session_data.get('user_name', 'Пользователь')
            city = session_data.get('address', '')
            
            print(f"Создание резюме для: {profession}, опыт: {experience[:50]}...")
            resume_text = await create_resume(profession, experience, additional, user_name, city)
            
            # Сбрасываем процесс
            session_data['current_process'] = None
            session_data['current_step'] = 0
            session_data['process_data'] = {}
            session_data['confirmed_data'] = {}

            # Добавляем в историю сообщений
            human_msg = HumanMessage(content=f"Создание резюме: {profession}, {experience}, {additional}")
            ai_msg = AIMessage(content=resume_text)
            session_data['messages'].extend([human_msg, ai_msg])
            
            # Ограничиваем историю сообщений
            limited_messages = limit_messages_history(session_data['messages'])
            session_data['messages'].clear()
            session_data['messages'].extend(limited_messages)
            
            return f"Готово! Вот ваше резюме:\n\n{resume_text}"
            
    except Exception as e:
        print(f"Error in handle_resume_process: {e}")
        # Сбрасываем процесс при ошибке
        session_data['current_process'] = None
        session_data['current_step'] = 0
        session_data['process_data'] = {}
        session_data['confirmed_data'] = {}
        return "Извините, произошла ошибка при создании резюме. Пожалуйста, попробуйте еще раз."

async def confirm_db_data_for_resume(user_input: str, session_data: dict) -> str:
    """Подтверждение данных из БД для создания резюме - возвращает только текст"""
    if not session_data.get('needs_confirmation'):
        # Первый вход - предлагаем подтвердить данные из БД
        db_profession = session_data.get('job_info', '').strip()
        user_name = session_data.get('user_name', '').strip()
        city = session_data.get('address', '').strip()
        
        confirmation_message = "На основании ваших данных в системе, я вижу:"
        if db_profession:
            confirmation_message += f"\n- Специальность: {db_profession}"
        if user_name:
            confirmation_message += f"\n- Имя: {user_name}"
        if city:
            confirmation_message += f"\n- Город: {city}"
            
        confirmation_message += "\n\nИспользовать эти данные для резюме? (да/нет)"
        
        session_data['needs_confirmation'] = True
        return confirmation_message
    else:
        # Обработка ответа на подтверждение
        if user_input.lower() in ['да', 'yes', 'конечно', 'ага']:
            # Сохраняем подтвержденные данные
            db_profession = session_data.get('job_info', '')
            if db_profession:
                session_data['process_data']['profession'] = db_profession
            
            session_data['confirmed_data']['resume_confirmed'] = True
            session_data['needs_confirmation'] = False
            
            # Если профессия подтверждена, переходим к следующему шагу
            if db_profession:
                session_data['current_step'] = 2
                return f"Отлично! Используем специальность: {db_profession}. Теперь опишите ваш опыт работы (продолжительность, проекты, обязанности):"
            else:
                session_data['current_step'] = 1
                return "Хорошо! Теперь укажите вашу специальность для резюме:"
                
        elif user_input.lower() in ['нет', 'no', 'не']:
            session_data['confirmed_data']['resume_confirmed'] = True
            session_data['needs_confirmation'] = False
            session_data['current_step'] = 1
            return "Хорошо! Пожалуйста, укажите вашу специальность для резюме:"
        else:
            return "Пожалуйста, ответьте 'да' или 'нет'. Использовать данные из системы для создания резюме?"

async def handle_job_search_process(user_input: str, session_data: dict, step: int, selected_uid: str) -> str:
    """Обрабатывает процесс поиска работы"""
    # Для зарегистрированных пользователей проверяем и подтверждаем данные из БД
    if selected_uid != "Без аккаунта" and not session_data.get('confirmed_data', {}).get('job_search_confirmed'):
        return await confirm_db_data_for_job_search(user_input, session_data)
    
    if step == 1:  # Шаг 1: Вакансия
        session_data['process_data']['vacancy'] = user_input
        session_data['current_step'] = 2
        return "Хорошо! Теперь укажите город для поиска работы:"
    
    elif step == 2:  # Шаг 2: Город
        session_data['process_data']['city'] = user_input
        
        # Выполняем поиск работы
        vacancy = session_data['process_data']['vacancy']
        city = session_data['process_data']['city']
        
        try:
            job_info = await extract_job_info(
                f"{vacancy} в {city}", 
                session_data['complete_data_str'], 
                session_data['selected_uid']
            )
            
            links = await main_jf_advanced(
                job_info,
                browser_type="firefox",
                use_cache=True,
                use_alternatives=True,
                debug=False,
                max_results=3
            )
            
            # Форматируем результат
            if links:
                prev_link = ""
                response = "Вот найденные вакансии:\n\n"
                for i, link in enumerate(links, 1):
                    if isinstance(link, dict):
                        if prev_link != link['url']:
                            response += f"{i}. **{link['title']}**\n"
                            response += f"   Ссылка:   <a href='{link['url']}' rel='noreferrer'> {link['url'][:14]}... </a>"
                            if link.get('salary'):
                                response += f"   Зарплата: {link['salary']}\n"
                            prev_link=link['url']
                    else:
                        response += f"{i}. {link}\n"
                    response += "\n"
            else:
                response = "К сожалению, по вашему запросу ничего не найдено. Попробуйте изменить параметры поиска."
            
        except Exception as e:
            response = f"Произошла ошибка при поиске вакансий: {str(e)}"
        
        # Сбрасываем процесс
        session_data['current_process'] = None
        session_data['current_step'] = 0
        session_data['process_data'] = {}
        session_data['confirmed_data'] = {}
        
        return response

async def confirm_db_data_for_job_search(user_input: str, session_data: dict) -> str:
    """Подтверждение данных из БД для поиска работы"""
    if not session_data.get('needs_confirmation'):
        # Первый вход - предлагаем подтвердить данные из БД
        db_profession = session_data.get('job_info', '').strip()
        db_city = session_data.get('address', '').strip()
        
        has_profession = bool(db_profession)
        has_city = bool(db_city)
        
        session_data['needs_confirmation'] = True
        
        if has_profession and has_city:
            return f"На основании ваших данных в системе, я вижу что вы ищете работу как: {db_profession} в городе {db_city}. Используем эти данные для поиска? (да/нет)"
        elif has_profession:
            return f"На основании ваших данных в системе, я вижу что вы ищете работу как: {db_profession}. Используем эту специальность для поиска? (да/нет)"
        elif has_city:
            return f"На основании ваших данных в системе, я вижу что вы из города: {db_city}. Используем этот город для поиска? (да/нет)"
        else:
            # Если данных нет в БД, переходим к обычному процессу
            session_data['confirmed_data']['job_search_confirmed'] = True
            session_data['current_step'] = 1
            return "Начнем поиск работы! Пожалуйста, укажите какую вакансию вы ищете:"
    
    # Обработка ответа на подтверждение
    if user_input.lower() in ['да', 'yes', 'конечно', 'ага']:
        db_profession = session_data.get('job_info', '')
        db_city = session_data.get('address', '')
        
        # Используем данные из БД, если они есть
        if db_profession:
            session_data['process_data']['vacancy'] = db_profession
        if db_city:
            session_data['process_data']['city'] = db_city
        
        session_data['confirmed_data']['job_search_confirmed'] = True
        session_data['needs_confirmation'] = False
        
        # Определяем следующий шаг
        if db_profession and not db_city:
            session_data['current_step'] = 2
            return f"Отлично! Используем специальность: {db_profession}. Теперь укажите город для поиска работы:"
        elif not db_profession and db_city:
            session_data['current_step'] = 1
            return f"Хорошо! Используем город: {db_city}. Теперь укажите какую вакансию вы ищете:"
        else:
            # Оба параметра есть, выполняем поиск сразу
            vacancy = session_data['process_data']['vacancy']
            city = session_data['process_data']['city']
            
            try:
                job_info = await extract_job_info(
                    f"{vacancy} в {city}", 
                    session_data['complete_data_str'], 
                    session_data['selected_uid']
                )
                
                links = await main_jf_advanced(
                    job_info,
                    browser_type="firefox",
                    use_cache=True,
                    use_alternatives=True,
                    debug=False,
                    max_results=3
                )
                
                # Форматируем результат
                if links:
                    response = "Вот найденные вакансии:\n\n"
                    for i, link in enumerate(links, 1):
                        if isinstance(link, dict):
                            response += f"{i}. **{link['title']}**\n"
                            response += f" <a href='{link['url']}' rel='noreferrer'> {link['url'][:14]}... </a>"
                            if link.get('salary'):
                                response += f"   Зарплата: {link['salary']}\n"
                        else:
                            response += f"{i}. {link}\n"
                        response += "\n"
                else:
                    response = "К сожалению, по вашему запросу ничего не найдено. Попробуйте изменить параметры поиска."
                
            except Exception as e:
                response = f"Произошла ошибка при поиске вакансий: {str(e)}"
            
            # Сбрасываем процесс
            session_data['current_process'] = None
            session_data['current_step'] = 0
            session_data['process_data'] = {}
            session_data['confirmed_data'] = {}
            
            return response
            
    elif user_input.lower() in ['нет', 'no', 'не']:
        session_data['confirmed_data']['job_search_confirmed'] = True
        session_data['needs_confirmation'] = False
        session_data['current_step'] = 1
        return "Хорошо! Начнем поиск работы заново. Пожалуйста, укажите какую вакансию вы ищете:"
    else:
        return "Пожалуйста, ответьте 'да' или 'нет'. Использовать данные из системы для поиска работы?"

async def handle_interview_process(user_input: str, session_data: dict, step: int, selected_uid: str) -> str:
    """Обрабатывает процесс подготовки к собеседованию"""
    # Для зарегистрированных пользователей проверяем и подтверждаем данные из БД
    if selected_uid != "Без аккаунта" and not session_data.get('confirmed_data', {}).get('interview_confirmed'):
        return await confirm_db_data_for_interview(user_input, session_data)
    
    if step == 1:  # Шаг 1: Вакансия
        session_data['process_data']['vacancy'] = user_input
        session_data['current_step'] = 2
        return "Отлично! Теперь укажите ваш примерный опыт работы в этой области (например: '1 год', '3 года', 'без опыта'):"
    
    elif step == 2:  # Шаг 2: Опыт
        session_data['process_data']['experience'] = user_input
        
        # Готовим вопросы для собеседования
        vacancy = session_data['process_data']['vacancy']
        experience = session_data['process_data']['experience']
        
        questions = await prepare_interview(vacancy, experience)
        
        # Сбрасываем процесс
        session_data['current_process'] = None
        session_data['current_step'] = 0
        session_data['process_data'] = {}
        session_data['confirmed_data'] = {}

        human_msg = HumanMessage(content=vacancy+", "+experience)
        ai_msg = AIMessage(content=questions)
        session_data['messages'].extend([human_msg, ai_msg])
        
        # Ограничиваем историю сообщений
        limited_messages = limit_messages_history(session_data['messages'])
        session_data['messages'].clear()
        session_data['messages'].extend(limited_messages)
        
        return f"Вот вопросы для подготовки к собеседованию на должность {vacancy}:\n\n{questions}"

async def confirm_db_data_for_interview(user_input: str, session_data: dict) -> str:
    """Подтверждение данных из БД для подготовки к собеседованию"""
    if not session_data.get('needs_confirmation'):
        # Первый вход - предлагаем подтвердить данные из БД
        db_profession = session_data.get('job_info', '').strip()
        if db_profession:
            session_data['needs_confirmation'] = True
            return f"На основании ваших данных в системе, я вижу что вы готовитесь к собеседованию на должность: {db_profession}. Используем эту вакансию для подготовки? (да/нет)"
        else:
            # Если данных нет в БД, переходим к обычному процессу
            session_data['confirmed_data']['interview_confirmed'] = True
            session_data['current_step'] = 1
            return "Начнем подготовку к собеседованию! Пожалуйста, укажите на какую должность вы готовитесь:"
    
    # Обработка ответа на подтверждение
    if user_input.lower() in ['да', 'yes', 'конечно', 'ага']:
        db_profession = session_data.get('job_info', '')
        session_data['process_data']['vacancy'] = db_profession
        session_data['confirmed_data']['interview_confirmed'] = True
        session_data['needs_confirmation'] = False
        session_data['current_step'] = 2
        return f"Отлично! Используем вакансию: {db_profession}. Теперь укажите ваш примерный опыт работы в этой области (например: '1 год', '3 года', 'без опыта'):"
    elif user_input.lower() in ['нет', 'no', 'не']:
        session_data['confirmed_data']['interview_confirmed'] = True
        session_data['needs_confirmation'] = False
        session_data['current_step'] = 1
        return "Хорошо! Пожалуйста, укажите на какую должность вы готовитесь:"
    else:
        return "Пожалуйста, ответьте 'да' или 'нет'. Использовать вакансию из ваших данных в системе для подготовки к собеседованию?"

async def handle_command(command: str, session_data: dict) -> Tuple[str, List[Dict]]:
    """Handle sidebar commands with buttons ONLY for completed responses"""
    print(f"Handling command: {command}")
    
    # Обработка команд для начала процессов
    if command == 'start_resume':
        session_data['current_process'] = 'resume'
        session_data['current_step'] = 1
        session_data['process_data'] = {}
        session_data['confirmed_data'] = {}
        session_data['needs_confirmation'] = False
        
        # Для зарегистрированных пользователей сразу начинаем подтверждение данных
        if session_data['selected_uid'] != "Без аккаунта":
            response_text = await confirm_db_data_for_resume("", session_data)
        else:
            response_text = "Начнем создание резюме! Пожалуйста, укажите вашу специальность:"
        
        # Во время активного процесса не показываем кнопки
        return response_text, []
    
    elif command == 'start_job_search':
        session_data['current_process'] = 'job_search'
        session_data['current_step'] = 1
        session_data['process_data'] = {}
        session_data['confirmed_data'] = {}
        session_data['needs_confirmation'] = False
        
        # Для зарегистрированных пользователей сразу начинаем подтверждение данных
        if session_data['selected_uid'] != "Без аккаунта":
            response_text = await confirm_db_data_for_job_search("", session_data)
        else:
            response_text = "Начнем поиск работы! Пожалуйста, укажите какую вакансию вы ищете:"
        
        return response_text, []
    
    elif command == 'start_interview_prep':
        session_data['current_process'] = 'interview'
        session_data['current_step'] = 1
        session_data['process_data'] = {}
        session_data['confirmed_data'] = {}
        session_data['needs_confirmation'] = False
        
        # Для зарегистрированных пользователей сразу начинаем подтверждение данных
        if session_data['selected_uid'] != "Без аккаунта":
            response_text = await confirm_db_data_for_interview("", session_data)
        else:
            response_text = "Начнем подготовку к собеседованию! Пожалуйста, укажите на какую должность вы готовитесь:"
        
        return response_text, []
    
    # Статические команды
    commands = {
        'about': '''Привет! Я знаю, что путь к финансовому благополучию может казаться сложным, но ты не один. Я здесь, чтобы быть твоим проводником и поддержкой.

Вместе мы сможем:
· 🎯 Найти крутую работу. Составим резюме, подготовимся к собеседованию, найдем подходящие вакансии.
· 📚 Научиться управлять деньгами. Объясню, как вести бюджет, как работают кредиты и инвестиции, как ставить финансовые цели.
· 📊 Улучшить твое финансовое положение. Проанализируем твои доходы и расходы, найдем способы сэкономить и начать копить.

Каждая большая цель начинается с маленького шага. Какой шаг ты готов сделать сегодня? Давай создадим твое успешное завтра, начав сегодня.''',
        'help': 'Вы можете использовать боковые кнопки для быстрого доступа к функций или просто общаться со мной в чате.',
        'examples': 'Примеры вопросов: "Помоги найти работу", "Составь резюме программиста", "Подготовь к собеседованию"'
    }
    
    response_text = commands.get(command, 'Команда не распознана.')
    
    # Обновляем историю сообщений для команд
    if command in commands:
        human_msg = HumanMessage(content=f"Команда: {command}")
        ai_msg = AIMessage(content=response_text)
        session_data['messages'].extend([human_msg, ai_msg])
        
        # Ограничиваем историю сообщений
        limited_messages = limit_messages_history(session_data['messages'])
        session_data['messages'].clear()
        session_data['messages'].extend(limited_messages)
    
    response_text = str(response_text) if response_text is not None else ""
    buttons = get_buttons_for_response(session_data, response_text, 'command')
    
    return response_text, buttons

async def handle_message(user_input: str, session_data: dict) -> Tuple[str, List[Dict]]:
    """Handle regular chat messages with buttons"""
    print(f"Handling message: {user_input}")
    
    # Проверяем, есть ли активный процесс сбора данных
    if session_data.get('current_process'):
        response_text = await handle_process_step(user_input, session_data, session_data['selected_uid'])
        # Во время активного процесса не показываем кнопки
        return response_text, []
    
    # Проверяем специальные запросы
    user_input_lower = user_input.lower()
    
    if user_input_lower in ['платежи', 'мои платежи', 'инфо о платежах']:
        response_text = await handle_payment_info_request(session_data)
    elif user_input_lower in ['баланс', 'мой баланс', 'средства']:
        response_text = await handle_balance_request(session_data)
    # Проверяем, не спрашивает ли пользователь о платежах
    elif any(keyword in user_input_lower for keyword in ['платеж', 'платить', 'сколько', 'сумма', 'деньги', 'оплата', 'задолженность']):
        response_text = await handle_payment_question(user_input, session_data)
    else:
        # Обычная обработка сообщения
        try:
            # Просто получаем ответ от модели без определения намерений
            reply, tokens = await get_response(
                session_data['complete_data_str'], 
                user_input, 
                session_data['messages'], 
                0
            )

            # Обновляем историю сообщений
            session_data['hist_r'] += f"human: {user_input}; ai: {reply}; "
            
            # Добавляем сообщения в историю и ограничиваем
            human_msg = HumanMessage(content=user_input)
            ai_msg = AIMessage(content=reply)
            session_data['messages'].extend([human_msg, ai_msg])
            
            # Ограничиваем историю сообщений
            limited_messages = limit_messages_history(session_data['messages'])
            session_data['messages'].clear()
            session_data['messages'].extend(limited_messages)
            
            response_text = reply
            
        except Exception as e:
            print(f"Error in handle_message: {e}")
            response_text = "Извините, произошла ошибка при обработке вашего сообщения."
    
    # Получаем кнопки для ответа
    buttons = get_buttons_for_response(session_data, response_text, 'message')
    
    return response_text, buttons

def parse_amount(amount_str: str) -> float:
    """Парсит строку с суммой платежа в число"""
    try:
        if not amount_str or amount_str.strip() == '':
            return 0.0
        
        # Убираем пробелы и заменяем запятую на точку
        cleaned = amount_str.replace(' ', '').replace(',', '.')
        return float(cleaned)
    except (ValueError, TypeError):
        print(f"Ошибка парсинга суммы: {amount_str}")
        return 0.0

def format_amount(amount: float) -> str:
    """Форматирует сумму для отображения"""
    return f"{amount:,.2f}".replace(',', ' ').replace('.', ',')

def calculate_total_payments(obligations: List[Dict]) -> float:
    """Рассчитывает общую сумму предстоящих платежей"""
    total = 0.0
    for obligation in obligations:
        payment_amount_str = obligation.get('payment_amount', '')
        if payment_amount_str:
            total += parse_amount(payment_amount_str)
    return total

def get_upcoming_payments(obligations: List[Dict]) -> str:
    """Возвращает строку с информацией о ближайших платежах"""
    upcoming = []
    today = datetime.now().date()
    
    for obligation in obligations:
        payment_date_str = obligation.get('payment_day', '')
        payment_amount_str = obligation.get('payment_amount', '')
        
        if payment_date_str and payment_amount_str:
            payment_date = parse_date(payment_date_str)
            if payment_date and payment_date.date() >= today:
                days_until = (payment_date.date() - today).days
                amount = parse_amount(payment_amount_str)
                upcoming.append({
                    'date': payment_date_str,
                    'amount': amount,
                    'days_until': days_until
                })
    
    # Сортируем по дате
    upcoming.sort(key=lambda x: x['days_until'])
    
    if not upcoming:
        return "Нет предстоящих платежей"
    
    result = []
    for payment in upcoming[:3]:  # Берем 3 ближайших платежа
        if payment['days_until'] == 0:
            result.append(f"сегодня {format_amount(payment['amount'])} руб.")
        elif payment['days_until'] == 1:
            result.append(f"завтра {format_amount(payment['amount'])} руб.")
        else:
            result.append(f"{payment['date']} {format_amount(payment['amount'])} руб.")
    
    return ", ".join(result)

async def handle_payment_question(user_input: str, session_data: dict) -> str:
    """Обрабатывает вопросы о платежах с учетом баланса и временных статусов"""
    if session_data['selected_uid'] == "Без аккаунта":
        return "Для получения информации о платежах необходимо выбрать пользователя из списка."
    
    obligations = session_data.get('obligations', [])
    if not obligations:
        return "У вас нет данных о платежах в системе."
    
    total_payments = session_data.get('total_payments', 0)
    account_balance = get_account_balance(obligations)
    
    # Получаем все платежи с временными статусами
    all_payments = get_all_payments_with_status(obligations)
    
    # Анализируем вопрос пользователя
    user_input_lower = user_input.lower()
    
    if 'сколько' in user_input_lower and 'платить' in user_input_lower:
        response = f"Общая сумма предстоящих платежей: {format_amount(total_payments)} руб.\n"
        response += f"Баланс на счетах: {format_amount(account_balance)} руб.\n\n"
        
        # Добавляем информацию о достаточности средств
        if account_balance <= 0:
            response += "⚠️ На ваших счетах нет средств! Срочно пополните счет.\n\n"
        elif account_balance < total_payments:
            shortage = total_payments - account_balance
            response += f"⚠️ Внимание! На счетах недостаточно средств. Не хватает: {format_amount(shortage)} руб.\n\n"
        else:
            response += "✅ На счетах достаточно средств для покрытия всех обязательств.\n\n"
        
        # Добавляем детализацию по каждому обязательству с временными статусами
        response += "Детализация по договорам:\n"
        for i, payment in enumerate(all_payments, 1):
            contract = payment['contract']
            payment_date = payment['payment_date']
            amount = payment['formatted_amount']
            status = payment['status']
            days_info = payment['days_info']
            
            # Выбираем иконку в зависимости от статуса
            if payment['status_type'] == 'overdue':
                status_icon = "🔴"
            elif payment['status_type'] == 'today':
                status_icon = "🟡"
            elif payment['status_type'] == 'upcoming':
                status_icon = "🟢"
            else:
                status_icon = "⚪"
            
            response += f"{i}. {status_icon} Договор {contract}: {amount} руб.\n"
            response += f"   📅 Срок: {payment_date} ({status})\n"
            if days_info:
                response += f"   ⏰ {days_info}\n"
            
            # Добавляем информацию о достаточности средств для этого платежа
            if account_balance < payment['amount_numeric']:
                shortage = payment['amount_numeric'] - account_balance
                response += f"   ⚠️ Не хватает: {format_amount(shortage)} руб.\n"
            response += "\n"
        
        return response
    
    elif 'когда' in user_input_lower and 'платить' in user_input_lower:
        response = "Ваши платежи:\n\n"
        
        # Группируем платежи по статусам
        overdue_payments = [p for p in all_payments if p['status_type'] == 'overdue']
        today_payments = [p for p in all_payments if p['status_type'] == 'today']
        upcoming_payments = [p for p in all_payments if p['status_type'] == 'upcoming']
        future_payments = [p for p in all_payments if p['status_type'] == 'future']
        
        # Просроченные платежи
        if overdue_payments:
            response += "🔴 ПРОСРОЧЕННЫЕ ПЛАТЕЖИ:\n"
            for payment in overdue_payments:
                response += f"• {payment['contract']}: {payment['formatted_amount']} руб. ({payment['payment_date']})\n"
                response += f"  {payment['days_info']}\n"
            response += "\n"
        
        # Платежи на сегодня
        if today_payments:
            response += "🟡 СЕГОДНЯ:\n"
            for payment in today_payments:
                response += f"• {payment['contract']}: {payment['formatted_amount']} руб.\n"
            response += "\n"
        
        # Ближайшие платежи (завтра и в течение недели)
        if upcoming_payments:
            response += "🟢 БЛИЖАЙШИЕ ПЛАТЕЖИ:\n"
            for payment in upcoming_payments:
                response += f"• {payment['contract']}: {payment['formatted_amount']} руб. ({payment['payment_date']})\n"
                response += f"  {payment['days_info']}\n"
            response += "\n"
        
        # Будущие платежи
        if future_payments:
            response += "⚪ БУДУЩИЕ ПЛАТЕЖИ:\n"
            for payment in future_payments[:3]:  # Показываем только 3 ближайших будущих
                response += f"• {payment['contract']}: {payment['formatted_amount']} руб. ({payment['payment_date']})\n"
                response += f"  {payment['days_info']}\n"
            if len(future_payments) > 3:
                response += f"  ... и еще {len(future_payments) - 3} платеж(а)\n"
            response += "\n"
        
        response += f"Баланс на счетах: {format_amount(account_balance)} руб.\n"
        
        if account_balance <= 0:
            response += "\n⚠️ На ваших счетах нет средств! Срочно пополните счет."
        elif account_balance < total_payments:
            shortage = total_payments - account_balance
            response += f"\n⚠️ Внимание! На счетах недостаточно средств. Не хватает: {format_amount(shortage)} руб."
        
        return response
    
    elif 'баланс' in user_input_lower or 'средств' in user_input_lower:
        response = f"💰 Баланс на ваших счетах: {format_amount(account_balance)} руб.\n"
        response += f"📊 Общая сумма предстоящих платежей: {format_amount(total_payments)} руб.\n\n"
        
        # Статус средств
        if account_balance <= 0:
            response += "🔴 КРИТИЧЕСКАЯ СИТУАЦИЯ:\n"
            response += "На ваших счетах нет средств! Срочно пополните счет."
        elif account_balance < total_payments:
            shortage = total_payments - account_balance
            response += "🟡 ВНИМАНИЕ:\n"
            response += f"На счетах недостаточно средств. Не хватает: {format_amount(shortage)} руб."
        else:
            response += "🟢 ОТЛИЧНО:\n"
            response += "На счетах достаточно средств для покрытия всех обязательств."
        
        # Добавляем информацию о ближайшем платеже
        upcoming_payments = [p for p in all_payments if p['status_type'] in ['today', 'upcoming']]
        if upcoming_payments:
            next_payment = upcoming_payments[0]
            response += f"\n\n📅 Ближайший платеж:\n"
            response += f"Договор {next_payment['contract']}: {next_payment['formatted_amount']} руб.\n"
            response += f"Срок: {next_payment['payment_date']} ({next_payment['status']})"
        
        return response
    
    elif 'просроч' in user_input_lower:
        overdue_payments = [p for p in all_payments if p['status_type'] == 'overdue']
        if overdue_payments:
            response = "🔴 ВАШИ ПРОСРОЧЕННЫЕ ПЛАТЕЖИ:\n\n"
            total_overdue = sum(p['amount_numeric'] for p in overdue_payments)
            
            for payment in overdue_payments:
                response += f"• Договор {payment['contract']}: {payment['formatted_amount']} руб.\n"
                response += f"  Просрочен: {payment['days_info']}\n"
                response += f"  Дата платежа: {payment['payment_date']}\n\n"
            
            response += f"📊 Общая сумма просрочки: {format_amount(total_overdue)} руб.\n"
            response += f"💰 Ваш баланс: {format_amount(account_balance)} руб.\n\n"
            
            if account_balance < total_overdue:
                shortage = total_overdue - account_balance
                response += f"⚠️ Для погашения просрочки не хватает: {format_amount(shortage)} руб."
            else:
                response += "✅ У вас достаточно средств для погашения просрочки."
        else:
            response = "✅ Отлично! У вас нет просроченных платежей."
        
        return response
    
    else:
        # Общая информация о платежах
        response = "📊 ОБЗОР ВАШИХ ПЛАТЕЖЕЙ\n\n"
        response += f"• Общая сумма к оплате: {format_amount(total_payments)} руб.\n"
        response += f"• Баланс на счетах: {format_amount(account_balance)} руб.\n"
        
        # Статистика по статусам
        overdue_count = len([p for p in all_payments if p['status_type'] == 'overdue'])
        today_count = len([p for p in all_payments if p['status_type'] == 'today'])
        upcoming_count = len([p for p in all_payments if p['status_type'] == 'upcoming'])
        
        response += f"• Статус платежей: "
        status_parts = []
        if overdue_count > 0:
            status_parts.append(f"🔴 {overdue_count} просрочено")
        if today_count > 0:
            status_parts.append(f"🟡 {today_count} сегодня")
        if upcoming_count > 0:
            status_parts.append(f"🟢 {upcoming_count} ближайшие")
        if status_parts:
            response += ", ".join(status_parts)
        else:
            response += "✅ Все платежи выполнены"
        
        response += "\n\n"
        
        if account_balance <= 0:
            response += "⚠️ На ваших счетах нет средств! Срочно пополните счет.\n\n"
        elif account_balance < total_payments:
            shortage = total_payments - account_balance
            response += f"⚠️ Внимание! На счетах недостаточно средств. Не хватает: {format_amount(shortage)} руб.\n\n"
        
        response += "Для получения подробной информации спросите:\n"
        response += "• 'Сколько мне нужно заплатить?' - полная детализация\n" 
        response += "• 'Когда мне платить?' - график платежей\n"
        response += "• 'Какой у меня баланс?' - статус средств\n"
        response += "• 'Какие просрочки?' - информация о просроченных платежах"
        
        return response

def get_all_payments_with_status(obligations: List[Dict]) -> List[Dict]:
    """Возвращает все платежи с временными статусами и информацией"""
    today = datetime.now().date()
    payments = []
    
    for obligation in obligations:
        payment_date_str = obligation.get('payment_day', '')
        payment_amount_str = obligation.get('payment_amount', '')
        
        if payment_date_str and payment_amount_str:
            payment_date = parse_date(payment_date_str)
            if payment_date:
                payment_date_date = payment_date.date()
                amount_numeric = parse_amount(payment_amount_str)
                formatted_amount = format_amount(amount_numeric)
                
                # Определяем статус платежа
                days_diff = (payment_date_date - today).days
                
                if days_diff < 0:
                    status_type = 'overdue'
                    days_passed = abs(days_diff)
                    if days_passed == 1:
                        status = "Просрочен на 1 день"
                        days_info = f"Просрочен вчера"
                    else:
                        status = f"Просрочен на {days_passed} дней"
                        days_info = f"Просрочен {days_passed} дней назад"
                        
                elif days_diff == 0:
                    status_type = 'today'
                    status = "Сегодня"
                    days_info = "Срок оплаты сегодня"
                    
                elif days_diff == 1:
                    status_type = 'upcoming'
                    status = "Завтра"
                    days_info = "Остался 1 день"
                    
                elif 2 <= days_diff <= 7:
                    status_type = 'upcoming'
                    status = "На этой неделе"
                    days_info = f"Осталось {days_diff} дней"
                    
                elif 8 <= days_diff <= 30:
                    status_type = 'future'
                    status = "В этом месяце"
                    days_info = f"Осталось {days_diff} дней"
                    
                else:
                    status_type = 'future'
                    status = "Будущий платеж"
                    days_info = f"Осталось {days_diff} дней"
                
                payments.append({
                    'contract': obligation.get('contract', 'Не указан'),
                    'payment_date': payment_date_str,
                    'amount_numeric': amount_numeric,
                    'formatted_amount': formatted_amount,
                    'status_type': status_type,
                    'status': status,
                    'days_info': days_info,
                    'days_until': days_diff,
                    'credit_product': obligation.get('credit_product', '')
                })
    
    # Сортируем платежи: сначала просроченные, затем по дате
    payments.sort(key=lambda x: (
        0 if x['status_type'] == 'overdue' else 
        1 if x['status_type'] == 'today' else
        2 if x['status_type'] == 'upcoming' else 3,
        x['days_until']
    ))
    
    return payments

def parse_date(date_string):
    """Парсит дату из формата 'дд.мм.гггг' в datetime объект"""
    try:
        if not date_string or date_string.strip() == '':
            return None
        return datetime.strptime(date_string.strip(), "%d.%m.%Y")
    except ValueError:
        print(f"Ошибка парсинга даты: {date_string}")
        return None

def calculate_days_until_payment(payment_date):
    """Вычисляет количество дней до платежа"""
    if not payment_date:
        return None
    today = datetime.now().date()
    payment_date_date = payment_date.date()
    return (payment_date_date - today).days

def get_account_balance(obligations: List[Dict]) -> float:
    """Получает баланс счета пользователя из обязательств"""
    for obligation in obligations:
        balance_str = obligation.get('account_balance', '')
        if balance_str and balance_str.strip():
            return parse_amount(balance_str)
    return 0.0

def get_funds_status_message(payment_amount: float, account_balance: float, contract: str) -> str:
    """Возвращает сообщение о статусе средств в зависимости от баланса"""
    if account_balance <= 0:
        return f"Срочно пополните счет! У вас нет средств для платежа по договору {contract}. Сумма платежа: {format_amount(payment_amount)} руб."
    elif account_balance < payment_amount:
        shortage = payment_amount - account_balance
        return f"Внимание! На вашем счете недостаточно средств для платежа по договору {contract}. Сумма платежа: {format_amount(payment_amount)} руб. Не хватает: {format_amount(shortage)} руб."
    else:
        return f"На вашем счете достаточно средств для платежа. Не забудьте внести платеж по договору {contract}. Сумма: {format_amount(payment_amount)} руб."

def calculate_grace_period_end(restruct_date_str):
    """Вычисляет дату окончания льготного периода (6 месяцев от даты реструктуризации)"""
    if not restruct_date_str:
        return None
    
    restruct_date = parse_date(restruct_date_str)
    if not restruct_date:
        return None
    
    # Добавляем 6 месяцев к дате реструктуризации
    grace_period_end = restruct_date + timedelta(days=6*30)  # Примерно 6 месяцев
    return grace_period_end

def get_grace_period_notification(restruct_date_str, payment_amount_str, contract):
    """Генерирует уведомление об окончании льготного периода"""
    print(f"DEBUG: Проверка ЛП для договора {contract}, дата реструктуризации: {restruct_date_str}")
    
    grace_period_end = calculate_grace_period_end(restruct_date_str)
    if not grace_period_end:
        print(f"DEBUG: Не удалось вычислить дату окончания ЛП для {contract}")
        return None
    
    today = datetime.now().date()
    grace_end_date = grace_period_end.date()
    
    print(f"DEBUG: Окончание ЛП: {grace_end_date}, сегодня: {today}")
    
    # Проверяем, что льготный период еще не закончился
    if grace_end_date <= today:
        print(f"DEBUG: ЛП уже закончился для {contract}")
        return None
    
    # Вычисляем разницу в днях до окончания ЛП
    days_remaining = (grace_end_date - today).days
    
    # Вычисляем приблизительное количество месяцев (округляем в большую сторону)
    months_remaining = (days_remaining + 29) // 30  # Округляем до ближайшего месяца
    
    print(f"DEBUG: Дней до окончания ЛП: {days_remaining}, месяцев: {months_remaining}")
    
    # Уведомляем за 2 месяца до окончания
    if months_remaining <= 3:
        payment_amount = parse_amount(payment_amount_str)
        formatted_amount = format_amount(payment_amount)
        
        # Дата следующего платежа после окончания ЛП
        next_payment_date = grace_end_date + timedelta(days=1)
        
        notification_data = {
            'id': f"grace_period_{contract}_{grace_end_date}",
            'type': 'grace_period',
            'title': '⚠️ Окончание льготного периода',
            'message': f"Информация об окончании ЛП - ежемесячно\nДо окончания Вашего ЛП осталось 2 месяца. Начиная с {next_payment_date.strftime('%d.%m.%Y')} сумма Вашего ежемесячного платежа будет составлять {formatted_amount} рублей",
            'grace_period_end': grace_end_date.strftime('%Y-%m-%d'),
            'grace_period_end_display': grace_end_date.strftime('%d.%m.%Y'),
            'months_remaining': months_remaining,
            'new_payment_amount': formatted_amount,
            'contract': contract,
            'read': False,
            'created_at': datetime.now().isoformat(),
            'priority': 'medium'
        }
        
        print(f"DEBUG: Создано уведомление об ЛП: {notification_data}")
        return notification_data
    
    print(f"DEBUG: Уведомление не создано - осталось {months_remaining} месяцев (требуется 2)")
    return None

async def generate_payment_notifications(user_id: str, session_data: dict):
    """Генерирует уведомления для ВСЕХ обязательств пользователя на 7 дней вперед с учетом баланса и льготного периода"""
    if user_id not in user_notifications:
        user_notifications[user_id] = []
    
    notifications = user_notifications[user_id]
    
    # Очищаем старые уведомления (теперь 60 дней для истории)
    current_time = datetime.now()
    notifications[:] = [n for n in notifications if 
                       (current_time - datetime.fromisoformat(n['created_at'])).days <= 60]
    
    # Получаем ВСЕ обязательства
    obligations = session_data.get('obligations', [])
    
    # print(f"DEBUG: Генерация уведомлений для пользователя {user_id}, обязательств: {len(obligations)}")
    
    # Получаем баланс счета пользователя
    account_balance = get_account_balance(obligations)
    
    # Собираем все ближайшие даты платежей с суммами
    payment_dates = []
    for obligation in obligations:
        payment_date_str = obligation.get('payment_day', '')
        payment_amount_str = obligation.get('payment_amount', '')
        restruct_date_str = obligation.get('restruct_date', '')
        
        if payment_date_str and payment_amount_str:
            payment_date = parse_date(payment_date_str)
            payment_amount = parse_amount(payment_amount_str)
            if payment_date:
                days_until = calculate_days_until_payment(payment_date)
                if days_until is not None:
                    payment_dates.append({
                        'date': payment_date,
                        'days_until': days_until,
                        'amount': payment_amount,
                        'formatted_amount': format_amount(payment_amount),
                        'obligation_data': obligation,
                        'contract': obligation.get('contract', 'Не указан'),
                        'payment_date_str': payment_date_str,
                        'restruct_date_str': restruct_date_str
                    })

    # Сортируем по ближайшей дате
    payment_dates.sort(key=lambda x: x['days_until'] if x['days_until'] >= 0 else float('inf'))
    
    # Создаем уведомления для всех обязательств в течение 7 дней
    for payment_info in payment_dates:
        payment_date = payment_info['date']
        days_until = payment_info['days_until']
        amount = payment_info['amount']
        formatted_amount = payment_info['formatted_amount']
        obligation = payment_info['obligation_data']
        contract = payment_info['contract']
        payment_date_str = payment_info['payment_date_str']
        restruct_date_str = payment_info['restruct_date_str']
        
        # Пропускаем если больше 7 дней и не просрочено
        if days_until > 7 and days_until >= 0:
            continue
            
        # Получаем сообщение о статусе средств с ДАТОЙ
        funds_message = get_funds_status_message_with_date(amount, account_balance, contract, payment_date_str, days_until)
        
        # Проверяем, нет ли уже уведомления для этой даты и обязательства
        existing_notification = next(
            (n for n in notifications 
             if n.get('type') == 'payment' 
             and n.get('payment_date') == payment_date.strftime('%Y-%m-%d')
             and n.get('obligation_id') == obligation.get('row_num')),
            None
        )
        
        if existing_notification:
            # Обновляем существующее уведомление если дни изменились
            if existing_notification.get('days_until') != days_until:
                existing_notification['days_until'] = days_until
                existing_notification['read'] = False  # Сбрасываем прочитанность при изменении
                existing_notification['created_at'] = datetime.now().isoformat()
                existing_notification['message'] = funds_message  # Обновляем сообщение
                
                # Обновляем приоритет в зависимости от статуса средств
                if days_until == 0:
                    existing_notification['title'] = '💳 Сегодня срок платежа'
                    if account_balance < amount:
                        existing_notification['priority'] = 'high'
                    else:
                        existing_notification['priority'] = 'medium'
                elif 1 <= days_until <= 3:
                    existing_notification['title'] = '💳 Ближайший платеж'
                    if account_balance < amount:
                        existing_notification['priority'] = 'high'
                    else:
                        existing_notification['priority'] = 'medium'
                elif 4 <= days_until <= 7:
                    existing_notification['title'] = '💳 Напоминание о платеже'
                    existing_notification['priority'] = 'low'
                elif days_until < 0:
                    existing_notification['title'] = '⚠️ Просроченный платеж'
                    existing_notification['priority'] = 'high'
            continue
            
        # Создаем новое уведомление с учетом баланса И ДАТОЙ
        if days_until == 0:
            notification = {
                'id': len(notifications) + 1,
                'type': 'payment',
                'title': '💳 Сегодня срок платежа',
                'message': funds_message,
                'payment_date': payment_date.strftime('%Y-%m-%d'),
                'payment_date_display': payment_date_str,
                'days_until': 0,
                'amount': formatted_amount,
                'contract': contract,
                'account_balance': format_amount(account_balance),
                'obligation_id': obligation.get('row_num'),
                'read': False,
                'created_at': datetime.now().isoformat(),
                'priority': 'high' if account_balance < amount else 'medium'
            }
            notifications.append(notification)
            
        elif 1 <= days_until <= 3:
            notification = {
                'id': len(notifications) + 1,
                'type': 'payment',
                'title': '💳 Ближайший платеж',
                'message': funds_message,
                'payment_date': payment_date.strftime('%Y-%m-%d'),
                'payment_date_display': payment_date_str,
                'days_until': days_until,
                'amount': formatted_amount,
                'contract': contract,
                'account_balance': format_amount(account_balance),
                'obligation_id': obligation.get('row_num'),
                'read': False,
                'created_at': datetime.now().isoformat(),
                'priority': 'high' if account_balance < amount else 'medium'
            }
            notifications.append(notification)
            
        elif 4 <= days_until <= 7:
            notification = {
                'id': len(notifications) + 1,
                'type': 'payment',
                'title': '💳 Напоминание о платеж',
                'message': funds_message,
                'payment_date': payment_date.strftime('%Y-%m-%d'),
                'payment_date_display': payment_date_str,
                'days_until': days_until,
                'amount': formatted_amount,
                'contract': contract,
                'account_balance': format_amount(account_balance),
                'obligation_id': obligation.get('row_num'),
                'read': False,
                'created_at': datetime.now().isoformat(),
                'priority': 'low'
            }
            notifications.append(notification)
            
        elif days_until < 0:
            days_overdue = abs(days_until)
            overdue_message = get_overdue_message_with_date(amount, account_balance, contract, payment_date_str, days_overdue)
            
            notification = {
                'id': len(notifications) + 1,
                'type': 'payment',
                'title': '⚠️ Просроченный платеж',
                'message': overdue_message,
                'payment_date': payment_date.strftime('%Y-%m-%d'),
                'payment_date_display': payment_date_str,
                'days_until': days_until,
                'amount': formatted_amount,
                'contract': contract,
                'account_balance': format_amount(account_balance),
                'obligation_id': obligation.get('row_num'),
                'read': False,
                'created_at': datetime.now().isoformat(),
                'priority': 'high'
            }
            notifications.append(notification)
    

    # Добавляем уведомления об окончании льготного периода
    grace_period_count = 0
    for obligation in obligations:
        restruct_date_str = obligation.get('restruct_date', '')
        payment_amount_str = obligation.get('payment_amount', '')
        contract = obligation.get('contract', 'Не указан')
        
        # print(f"DEBUG: Проверка ЛП для {contract}: restruct_date={restruct_date_str}, payment={payment_amount_str}")
        
        if restruct_date_str and restruct_date_str.strip() and payment_amount_str and payment_amount_str.strip():
            grace_period_notification = get_grace_period_notification(restruct_date_str, payment_amount_str, contract)
            
            if grace_period_notification:
                # Проверяем, нет ли уже такого уведомления
                existing_grace_notification = next(
                    (n for n in notifications 
                     if n.get('type') == 'grace_period' 
                     and n.get('contract') == contract
                     and n.get('grace_period_end') == grace_period_notification['grace_period_end']),
                    None
                )
                
                if not existing_grace_notification:
                    notifications.append(grace_period_notification)
                    grace_period_count += 1
                    print(f"DEBUG: Добавлено уведомление об ЛП для {contract}")
                else:
                    print(f"DEBUG: Уведомление об ЛП для {contract} уже существует")
            else:
                print(f"DEBUG: Уведомление об ЛП для {contract} не создано")
        else:
            print(f"DEBUG: Пропущено ЛП для {contract} - отсутствуют данные: restruct_date={restruct_date_str}, payment={payment_amount_str}")
    
    # print(f"DEBUG: Добавлено уведомлений об ЛП: {grace_period_count}")
    
    # Сортируем по приоритету и дате (новые сначала)
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    notifications.sort(key=lambda x: (
        priority_order.get(x.get('priority', 'low'), 2),
        -datetime.fromisoformat(x['created_at']).timestamp()
    ))
    
    # Ограничиваем количество (последние 50)
    user_notifications[user_id] = notifications[-50:]
    
    # print(f"DEBUG: Итоговое количество уведомлений: {len(notifications)}")

def get_funds_status_message_with_date(payment_amount: float, account_balance: float, contract: str, payment_date: str, days_until: int) -> str:
    """Возвращает сообщение о статусе средств с указанием даты платежа"""
    days_text = ""
    if days_until == 0:
        days_text = "СЕГОДНЯ"
    elif days_until == 1:
        days_text = "ЗАВТРА"
    elif days_until > 1:
        days_text = f"через {days_until} дня(ей)"
    else:
        days_text = f"просрочен на {abs(days_until)} дня(ей)"
    
    base_message = f"Платеж по договору {contract} на сумму {format_amount(payment_amount)} руб. {days_text} ({payment_date})"
    
    if account_balance <= 0:
        return f"{base_message}. Срочно пополните счет!\n<a href='https://www.sberbank.ru/ru/person/credits/money/podderzhi'> Помощь близкому</a>"
    elif account_balance < payment_amount:
        shortage = payment_amount - account_balance
        return f"""{base_message}. На счете недостаточно средств. Не хватает: {format_amount(shortage)} руб. \n<a href='https://www.sberbank.ru/ru/person/credits/money/podderzhi'> Помощь близкому</a>"""
    else:
        return f"{base_message}. На счете достаточно средств."

def get_overdue_message_with_date(payment_amount: float, account_balance: float, contract: str, payment_date: str, days_overdue: int) -> str:
    """Возвращает сообщение о просроченном платеже с указанием даты"""
    base_message = f"Платеж по договору {contract} на {payment_date} просрочен на {days_overdue} дня(ей). Сумма: {format_amount(payment_amount)} руб."
    
    if account_balance <= 0:
        return f"{base_message}. Срочно пополните счет!"
    elif account_balance < payment_amount:
        shortage = payment_amount - account_balance
        return f"{base_message}. На счете недостаточно средств. Не хватает: {format_amount(shortage)} руб."
    else:
        return base_message

async def background_notification_updater():
    """Улучшенная фоновая задача для регулярного обновления уведомлений"""
    while True:
        try:
            # print("Background: Updating notifications for all users...")
            current_time = datetime.now()
            
            # Обновляем уведомления для всех активных пользователей
            for session_id, session_data in list(user_sessions.items()):
                user_id = session_data.get('selected_uid')
                
                if user_id and user_id != "Без аккаунта":
                    # Проверяем активность сессии (последние 10 минут)
                    if (current_time - session_data['last_activity']).total_seconds() < 600:
                        await generate_payment_notifications(user_id, session_data)
                        # print(f"Background: Updated notifications for active user {user_id}")
            
            await asyncio.sleep(100)
            
        except Exception as e:
            print(f"Error in background_notification_updater: {e}")
            await asyncio.sleep(20)  # Ждем 30 секунд при ошибке
            
def initialize_user_session(session_id, user_id):
    """Initialize session data for a user with ALL obligations"""
    print(f"Initializing session for user: {user_id}")
    
    if user_id != "Без аккаунта":
        # Используем автоматическое определение пути к CSV
        obligations, last_obligation = parse_csv_table('show_case_database.csv', user_id, delimiter=';', encoding='cp1251')
        
        # ... остальной код функции БЕЗ ИЗМЕНЕНИЙ ...
        total_payments = calculate_total_payments(obligations)
        upcoming_payments = get_upcoming_payments(obligations)
        
        complete_data_str = f"Все обязательства пользователя: {obligations}" if obligations else "Данные не найдены"
        
        system_message_text = f"""Ты - чат-бот, отвечающий на вопросы, как консультант после реструктуризации кредита/ипотеки.
Вот данные о клиенте:{complete_data_str}. 

Дополнительная финансовая информация:
- Общая сумма предстоящих платежей: {format_amount(total_payments)} руб.
- Ближайшие платежи: {upcoming_payments}

Если клиент хочет уточнить данные о своей задолженности или узнать своё состояние, используй эту информацию. Проанализируй из имеющийхся данных краткую сводку, которую ты сообщишь напрямую пользователю. Так же можешь уточнить, что используя кнопку "Ваш Финансовый анализ" пользователь получит более точные данные. Обращайся к клиенту по имени-отчеству. Твоя задача - помочь ему восстановиться, помочь как можно быстрее закрыть задолженность. Учти, что клиент уже прошел реструктуризацию, либо сразу после её проведения.

Если тема не имеет ничего общего с попытками восстановиться, то можешь отказать пользователю в запросе. Если считаешь что пользователю стоит воспользоваться функциями, которые указаны далее, то задай уточняющий вопрос, а так же сообщи об этом обязательно, и как можно выше в сообщении. Всё равно расскажи об этом, пользователь может использовать все эти функции в левой части экрана.
 - Поиск работы (прямые запросы: "найти вакансии", "ищу работу")
 - Резюме ("составить резюме", "помоги с резюме")
 - Собеседование ("подготовить к собеседованию", "опрос как на интервью")
 - Финансы ("Финансовый анализ", "финансовая ситуация")
 - Центр занятости ("биржа труда", "обратиться в цзн")
 - Самозанятость ("открыть ип", "работа на себя")
 - Соцконтракт ("социальный контракт", "господдержка малоимущим")
 - Декрет ("декретный отпуск", "уход за ребенком")
 - Больничный ("длительное лечение", "нетрудоспособность")
 - Служба ("призыв в армию", "срочная служба")

Если пользователь откажет, то просто веди с ним дальше диалог.

По возможности общайся с пользователем просто, старайся не использовать сложные формулировки или термины. Отвечай кратко, при необходимости можешь расширять сови ответы, так же можешь уточнять всё ли понятно пользователю.
"""
        
    else:
        obligations = []
        last_obligation = {}
        complete_data_str = ""
        system_message_text = """Ты - чат-бот, отвечающий на вопросы, как консультант после реструктуризации кредита/ипотеки.
Твоя задача - помочь ему восстановиться, помочь как можно быстрее закрыть задолженность. Учти, что клиент уже прошел реструктуризацию, либо сразу после её проведения.
Если тема не имеет ничего общего с попытками востановится, то можешь отказать пользователю в запросе.Если считаешь что пользователю стоит воспользоваться функциями, которые указаны далее, то задай уточняющий вопрос.
 - Поиск работы (прямые запросы: "найти вакансии", "ищу работу")
 - Резюме ("составить резюме", "помоги с резюме")
 - Собеседование ("подготовить к собеседованию", "опрос как на интервью")
 - Финансы ("Финансовый анализ", "финансовая ситуация")
 - Центр занятости ("биржа труда", "обратиться в цзн")
 - Самозанятость ("открыть ип", "работа на себя")
 - Соцконтракт ("социальный контракт", "господдержка малоимущим")
 - Декрет ("декретный отпуск", "уход за ребенком")
 - Больничный ("длительное лечение", "нетрудоспособность")
 - Служба ("призыв в армию", "срочная служба")

Если пользователь откажет, то просто веди с ним дальше диалог.

По возможности общайся с пользователем просто, старайся не использовать сложные формулировки или термины. Отвечай кратко, так же можешь уточнять всё ли понятно пользователю.
"""

    user_sessions[session_id] = {
        'selected_uid': user_id,
        'messages': [SystemMessage(content=system_message_text)],
        'hist_r': "",
        'obligations': obligations,
        'current_obligation': last_obligation,
        'complete_data_str': complete_data_str,
        'user_name': last_obligation.get('user_name', ''),
        'address': last_obligation.get('address', ''),
        'job_info': last_obligation.get('job_info', ''),
        'payment_day': last_obligation.get('payment_day', ''),
        'payment_amount': last_obligation.get('payment_amount', ''),
        'total_payments': calculate_total_payments(obligations),
        'upcoming_payments': get_upcoming_payments(obligations),
        'current_process': None,
        'process_data': {},
        'current_step': 0,
        'needs_confirmation': False,
        'confirmed_data': {},
        'last_activity': datetime.now()
    }
    
    debug_session(session_id, "initialized")
    return user_sessions[session_id]

def get_buttons_for_response(session_data: dict, response_text: str, message_type: str = None) -> List[Dict[str, Any]]:
    """
    Определяет, какие кнопки показывать после ответа бота
    Возвращает список объектов кнопок, а не их ID
    """
    print(f"DEBUG: get_buttons_for_response - message_type: {message_type}")
    
    # Если есть активный процесс, не показываем кнопки
    if session_data.get('current_process') and session_data.get('current_step', 0) > 0:
        return []
    
    # Определяем группу кнопок по контексту
    response_lower = str(response_text).lower()
    
    # Определяем группу кнопок по контексту
    if any(word in response_lower for word in ['финанс', 'деньг', 'платеж', 'кредит', 'долг', 'баланс']):
        button_ids = ['financial_analysis', 'payment_info', 'balance', 'self_employment', 'tax_deduction']
    elif any(word in response_lower for word in ['работа', 'ваканс', 'резюме', 'собеседован', 'карьер', 'труд']):
        button_ids = ['resume', 'job_search', 'interview', 'employment_center']
    elif any(word in response_lower for word in ['социал', 'государств', 'поддержк', 'пособи', 'контракт', 'нуждающ', 'малоимущ']):
        button_ids = ['self_employment', 'social_contract', 'employment_center', 'tax_deduction']
    elif any(word in response_lower for word in ['здоровь', 'медицин', 'врач', 'лечен', 'больничн', 'болезн']):
        button_ids = ['sber_health', 'sber_insurance', 'help_relative', 'sick_leave']
    elif any(word in response_lower for word in ['отпуск', 'декрет', 'ребенок', 'рождени', 'материнств']):
        button_ids = ['maternity_leave', 'help_relative', 'tax_deduction', 'social_contract']
    elif any(word in response_lower for word in ['служба', 'арми', 'военн', 'призыв']):
        button_ids = ['military_service', 'help_relative', 'employment_center']
    elif any(word in response_lower for word in ['страхован', 'полис', 'страховк', 'защит']):
        button_ids = ['sber_insurance', 'sber_health', 'financial_analysis']
    elif any(word in response_lower for word in ['юридич', 'право', 'консультац', 'закон', 'спор']):
        button_ids = ['sber_law', 'legal_advice', 'social_contract']
    elif any(word in response_lower for word in ['сервис', 'прайм', 'подписк', 'скидк', 'доставк']):
        button_ids = ['sber_prime', 'sber_health', 'sber_law']
    elif any(word in response_lower for word in ['помощь', 'близк', 'родственник', 'пожил']):
        button_ids = ['help_relative', 'sber_health', 'sber_insurance']
    elif message_type == 'process_complete':
        # После завершения процесса показываем основные кнопки
        button_ids = ['resume', 'job_search', 'interview', 'financial_analysis']
    else:
        # Кнопки по умолчанию
        button_ids = ['resume', 'job_search', 'interview', 'financial_analysis']

    # Для зарегистрированных пользователей добавляем финансовые кнопки
    user_id = session_data.get('selected_uid', 'Без аккаунта')
    if user_id != "Без аккаунта" and session_data.get('obligations'):
        if 'payment_info' not in button_ids:
            button_ids = ['payment_info'] + button_ids

    # Преобразуем ID в объекты кнопок
    button_templates = {
        'financial_analysis': {'text': "💰 Финансовый анализ", 'action': "startFinancialAnalysis()"},
        'payment_info': {'text': "💳 Инфо о платежах", 'action': "sendPaymentInfoRequest()"},
        'balance': {'text': "📊 Баланс", 'action': "sendBalanceRequest()"},
        'resume': {'text': "📝 Создать резюме", 'action': "startResumeCreation()"},
        'job_search': {'text': "🔍 Поиск работы", 'action': "startJobSearch()"},
        'interview': {'text': "💼 Подготовка к собеседованию", 'action': "startInterviewPrep()"},
        'employment_center': {'text': "🏢 Центр занятости", 'action': "showFinancialTopic(6)"},
        'self_employment': {'text': "💼 Самозанятость", 'action': "showFinancialTopic(1)"},
        'social_contract': {'text': "📄 Соц. контракт", 'action': "showFinancialTopic(2)"},
        'tax_deduction': {'text': "🧾 Налоговый вычет", 'action': "showFinancialTopic(14)"},
        'sber_health': {'text': "🏥 СберЗдоровье", 'action': "showFinancialTopic(7)"},
        'sber_insurance': {'text': "🛡️ СберСтрахование", 'action': "showFinancialTopic(12)"},
        'sber_law': {'text': "⚖️ СберПраво", 'action': "showFinancialTopic(9)"},
        'sber_prime': {'text': "👑 СберПрайм", 'action': "showFinancialTopic(8)"},
        'help_relative': {'text': "👵 Помощь близкому", 'action': "showFinancialTopic(10)"},
        'maternity_leave': {'text': "👶 Декретный отпуск", 'action': "showFinancialTopic(3)"},
        'sick_leave': {'text': "🏥 Длительный больничный", 'action': "showFinancialTopic(4)"},
        'military_service': {'text': "🎖️ Срочная служба", 'action': "showFinancialTopic(5)"},
        'legal_advice': {'text': "📝 Юридическая помощь", 'action': "showFinancialTopic(9)"},
    }
    
    buttons = []
    for button_id in button_ids[:8]:  # Ограничиваем количество
        if button_id in button_templates:
            buttons.append(button_templates[button_id])
    
    print(f"DEBUG: Returning {len(buttons)} buttons: {[b['text'] for b in buttons]}")
    return buttons

# Вспомогательные функции для обработки платежных запросов
async def handle_payment_info_request(session_data: dict) -> str:
    """Обрабатывает запрос информации о платежах"""
    return await handle_payment_question("сколько мне нужно заплатить", session_data)

async def handle_balance_request(session_data: dict) -> str:
    """Обрабатывает запрос информации о балансе"""
    return await handle_payment_question("какой у меня баланс", session_data)



# Маршруты Quart (остаются без изменений)
@app.route('/')
async def serve_html():
    """Serve the main HTML page"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_file = os.path.join(current_dir, 'dunno_v3.html')
        
        print(f"Текущая директория: {current_dir}")
        print(f"Ищем HTML файл: {html_file}")
        print(f"Файл существует: {os.path.exists(html_file)}")
        
        if not os.path.exists(html_file):
            # Покажем все файлы в директории для отладки
            files = os.listdir(current_dir)
            print(f"Файлы в директории: {files}")
            return f"HTML файл не найден. Файлы в директории: {files}", 500
            
        return await send_file(html_file)
    
    except Exception as e:
        return f"Ошибка загрузки HTML: {str(e)}", 500

@app.route('/api/users', methods=['GET'])
async def get_users():
    """Get available user IDs"""
    return jsonify({
        'status': 'success',
        'users': AVAILABLE_USER_IDS
    })

@app.route('/api/select-user', methods=['POST'])
async def select_user():
    """Select a user ID for the session"""
    try:
        data = await request.get_json()
        user_id = data.get('user_id', 'Без аккаунта')
        
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        session_data = initialize_user_session(session_id, user_id)
        
        if user_id != "Без аккаунта":
            await generate_payment_notifications(user_id, session_data)
        
        notifications = user_notifications.get(user_id, [])
        
        return jsonify({
            'status': 'success',
            'message': f'User {user_id} selected',
            'user_data_available': user_id != "Без аккаунта",
            'session_id': session_id,
            'notifications': notifications,
            'notifications_count': len(notifications)
        })
    
    except Exception as e:
        print(f"Error in select_user: {e}")
        return jsonify({
            'status': 'error',
            'response': f'Произошла ошибка: {str(e)}'
        }), 500

    
@app.route('/api/reset-session', methods=['POST'])
async def reset_session():
    """Reset user session completely"""
    try:
        session_id = session.get('session_id')
        user_id = session.get('selected_user', 'Без аккаунта')
        
        if session_id and session_id in user_sessions:
            # Удаляем старую сессию
            del user_sessions[session_id]
            print(f"Session {session_id} reset for user {user_id}")
        
        # Создаем новую сессию
        new_session_id = str(uuid.uuid4())
        session['session_id'] = new_session_id
        session['needs_refresh'] = True
        
        # Инициализируем новую сессию
        initialize_user_session(new_session_id, user_id)
        
        return jsonify({
            'status': 'success',
            'message': 'Session reset successfully',
            'new_session_id': new_session_id,
            'redirect': True
        })
    
    except Exception as e:
        print(f"Error in reset_session: {e}")
        return jsonify({'status': 'error'})

@app.route('/api/select-user-redirect', methods=['POST'])
async def select_user_redirect():
    """Select user and redirect to refresh page"""
    try:
        data = await request.get_json()
        user_id = data.get('user_id', 'Без аккаунта')
        
        # Сохраняем выбранного пользователя в сессии
        session['selected_user'] = user_id
        session['needs_refresh'] = True
        
        # Создаем сессию если её нет
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        # Инициализируем сессию
        session_data = initialize_user_session(session_id, user_id)
        
        # Генерируем уведомления
        if user_id != "Без аккаунта":
            await generate_payment_notifications(user_id, session_data)
        
        return jsonify({
            'status': 'success', 
            'redirect_url': '/refresh',
            'message': f'User {user_id} selected - refreshing page'
        })
    
    except Exception as e:
        print(f"Error in select_user_redirect: {e}")
        return jsonify({
            'status': 'error',
            'response': f'Произошла ошибка: {str(e)}'
        }), 500

@app.route('/api/chat', methods=['POST'])
async def chat_api():
    """Main chat endpoint with buttons ONLY for completed responses"""
    try:
        data = await request.get_json()
        user_input = data.get('content', '').strip()
        message_type = data.get('type', 'message')
        
        print(f"Received chat request: {user_input[:100]}...")
        
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        session_data = get_user_session(session_id)
        
        if not user_input:
            return jsonify({
                'status': 'error',
                'response': 'Пустое сообщение',
                'buttons': []
            })
        
        if message_type == 'command':
            response_text, buttons = await handle_command(user_input, session_data)
        else:
            response_text, buttons = await handle_message(user_input, session_data)
        
        return jsonify({
            'status': 'success',
            'response': response_text,
            'buttons': buttons,  # Кнопки будут только когда нет активного процесса
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in chat_api: {e}")
        return jsonify({
            'status': 'error',
            'response': f'Произошла ошибка: {str(e)}',
            'buttons': []
        }), 500

@app.route('/refresh')
async def refresh_page():
    """Refresh the page and reinitialize user session"""
    try:
        # Получаем текущего пользователя из сессии
        selected_user = session.get('selected_user', 'Без аккаунта')
        session_id = session.get('session_id')
        
        if session_id:
            # Переинициализируем сессию
            session_data = initialize_user_session(session_id, selected_user)
            
            # Генерируем уведомления
            if selected_user != "Без аккаунта":
                await generate_payment_notifications(selected_user, session_data)
        
        # Перенаправляем на главную страницу
        return await send_file("dunno_v3.html")  # Убедитесь, что путь правильный для Netlify
    
    except Exception as e:
        return f"Ошибка перезагрузки: {str(e)}", 500

@app.route('/api/current-user', methods=['GET'])
async def get_current_user():
    """Get current user information"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'status': 'success',
                'user_id': 'Без аккаунта',
                'user_data_available': False
            })
        
        session_data = get_user_session(session_id)
        user_id = session_data['selected_uid']
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'user_data_available': user_id != "Без аккаунта",
            'user_name': session_data.get('user_name', ''),
            'notifications_count': len(user_notifications.get(user_id, []))
        })
    
    except Exception as e:
        print(f"Error in get_current_user: {e}")
        return jsonify({'status': 'error'})

# Остальные маршруты API остаются без изменений
@app.route('/api/job-search', methods=['POST']) 
async def job_search_api():
    """Job search with multiple results"""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'response': 'Отсутствуют данные запроса'
            }), 400
            
        user_input = data.get('description', '').strip()
        if not user_input:
            return jsonify({
                'status': 'error', 
                'response': 'Не указано описание для поиска работы'
            }), 400
            
        max_results = min(data.get('max_results', 3), 10)  # Ограничиваем максимум 10 результатов
        
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'status': 'error',
                'response': 'Сначала выберите пользователя'
            }), 400
        
        session_data = get_user_session(session_id)
        
        # Добавляем таймаут для поиска работы
        try:
            job_info = await asyncio.wait_for(
                extract_job_info(
                    user_input, 
                    session_data['complete_data_str'], 
                    session_data['selected_uid']
                ),
                timeout=30.0
            )
            
            links = await asyncio.wait_for(
                main_jf_advanced(
                    job_info,
                    browser_type="firefox",
                    use_cache=True,
                    use_alternatives=True,
                    debug=False,
                    max_results=1
                ),
                timeout=60.0
            )
            
        except asyncio.TimeoutError:
            return jsonify({
                'status': 'error',
                'response': 'Поиск занял слишком много времени. Пожалуйста, попробуйте позже.'
            }), 408
            
        # Остальная логика обработки результатов...
        return jsonify({
            'status': 'success',
            'response': 'Результаты поиска работы'
        })
        
    except Exception as e:
        logging.error(f"Ошибка в job-search API: {e}")
        return jsonify({
            'status': 'error',
            'response': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/financial-analysis', methods=['POST'])
async def financial_analysis_api():
    """Financial analysis functionality with buttons"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'status': 'error',
                'response': 'Сначала выберите пользователя',
                'buttons': []
            })
        
        session_data = get_user_session(session_id)
        
        if session_data['selected_uid'] == "Без аккаунта":
            return jsonify({
                'status': 'error',
                'response': 'Эта функция доступна только для зарегистрированных пользователей',
                'buttons': []
            })
        
        # Используем данные из сессии для анализа
        analysis = await prepare_fin_analysis(session_data['obligations'], session_data)
        
        buttons = get_buttons_for_response(session_data, analysis, 'financial_analysis')
        
        return jsonify({
            'status': 'success',
            'response': analysis,
            'buttons': buttons
        })
    
    except Exception as e:
        print(f"Error in financial_analysis_api: {e}")
        return jsonify({
            'status': 'error',
            'response': f'Ошибка при анализе: {str(e)}',
            'buttons': []
        }), 500

@app.route('/api/financial-resources', methods=['GET'])
async def financial_resources_api():
    """Get financial resources"""
    return jsonify({
        'status': 'success',
        'resources': FINANCIAL_RESOURCES
    })    

@app.route('/api/notifications', methods=['GET'])
async def get_notifications():
    """Get notifications for current user with auto-refresh"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'status': 'success', 'notifications': []})
        
        session_data = get_user_session(session_id)
        user_id = session_data['selected_uid']
        
        if user_id == "Без аккаунта":
            return jsonify({'status': 'success', 'notifications': []})
        
        # Всегда генерируем актуальные уведомления при запросе
        await generate_payment_notifications(user_id, session_data)
        
        # Получаем уведомления для пользователя
        notifications = user_notifications.get(user_id, [])
        
        return jsonify({
            'status': 'success',
            'notifications': notifications,
            'auto_refreshed': True
        })
    
    except Exception as e:
        print(f"Error in get_notifications: {e}")
        return jsonify({'status': 'error', 'notifications': []})

@app.route('/api/refresh-notifications', methods=['POST'])
async def refresh_notifications():
    """Принудительное обновление уведомлений при смене пользователя"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'status': 'success', 'notifications': []})
        
        session_data = get_user_session(session_id)
        user_id = session_data['selected_uid']
        
        if user_id == "Без аккаунта":
            return jsonify({'status': 'success', 'notifications': []})
        
        # Генерируем актуальные уведомления
        await generate_payment_notifications(user_id, session_data)
        
        # Получаем обновленные уведомления
        notifications = user_notifications.get(user_id, [])
        
        return jsonify({
            'status': 'success',
            'notifications': notifications,
            'message': 'Уведомления обновлены'
        })
    
    except Exception as e:
        print(f"Error in refresh_notifications: {e}")
        return jsonify({'status': 'error', 'notifications': []})

@app.route('/api/notifications/read', methods=['POST'])
async def mark_notifications_read():
    """Mark all notifications as read"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'status': 'success'})
        
        session_data = get_user_session(session_id)
        user_id = session_data['selected_uid']
        
        if user_id in user_notifications:
            for notification in user_notifications[user_id]:
                notification['read'] = True
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        print(f"Error in mark_notifications_read: {e}")
        return jsonify({'status': 'error'})

@app.route('/api/payment-info', methods=['GET'])
async def get_payment_info():
    """Get detailed payment information for current user"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'status': 'error', 'message': 'No session'})
        
        session_data = get_user_session(session_id)
        user_id = session_data['selected_uid']
        
        if user_id == "Без аккаунта":
            return jsonify({'status': 'error', 'message': 'Эта функция доступна только для зарегистрированных пользователей'})
        
        obligations = session_data.get('obligations', [])
        total_payments = session_data.get('total_payments', 0)
        upcoming_payments = session_data.get('upcoming_payments', '')
        
        # Формируем детальную информацию по платежам
        payment_details = []
        for oblig in obligations:
            payment_date = oblig.get('payment_day', '')
            payment_amount = oblig.get('payment_amount', '')
            contract = oblig.get('contract', '')
            credit_product = oblig.get('credit_product', '')
            
            if payment_date and payment_amount:
                payment_details.append({
                    'contract': contract,
                    'credit_product': credit_product,
                    'payment_date': payment_date,
                    'payment_amount': payment_amount,
                    'amount_numeric': parse_amount(payment_amount),
                    'formatted_amount': format_amount(parse_amount(payment_amount))
                })
        
        return jsonify({
            'status': 'success',
            'total_payments': total_payments,
            'formatted_total': format_amount(total_payments),
            'upcoming_payments': upcoming_payments,
            'payment_details': payment_details,
            'obligations_count': len(obligations)
        })
    
    except Exception as e:
        print(f"Error in get_payment_info: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/notifications/<int:notification_id>/read', methods=['POST'])
async def mark_single_notification_read(notification_id):
    """Mark a single notification as read"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'status': 'success'})
        
        session_data = get_user_session(session_id)
        user_id = session_data['selected_uid']
        
        if user_id in user_notifications:
            for notification in user_notifications[user_id]:
                if notification['id'] == notification_id:
                    notification['read'] = True
                    break
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        print(f"Error in mark_single_notification_read: {e}")
        return jsonify({'status': 'error'})

@app.route('/api/debug-sessions', methods=['GET'])
async def debug_sessions():
    """Debug endpoint to check session state"""
    return jsonify({
        'status': 'success',
        'sessions_count': len(user_sessions),
        'sessions': list(user_sessions.keys())
    })

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.getLogger("httpx").setLevel(logging.WARNING)
    print("Сервер запускается...")
    print("Откройте в браузере: http://localhost:5000")
    
    # Проверяем доступность CSV файла
    csv_path = get_csv_path()
    if os.path.exists(csv_path):
        print(f"CSV файл найден: {csv_path}")
    else:
        print(f"ВНИМАНИЕ: CSV файл не найден по пути: {csv_path}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)