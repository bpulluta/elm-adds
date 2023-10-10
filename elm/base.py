# -*- coding: utf-8 -*-
"""
ELM abstract class for API calls
"""
from abc import ABC
import asyncio
import aiohttp
import openai
import requests
import tiktoken
import time
import logging


logger = logging.getLogger(__name__)


class ApiBase(ABC):
    """Class to parse text from a PDF document."""

    DEFAULT_MODEL = 'gpt-3.5-turbo'
    """Default model to do pdf text cleaning."""

    EMBEDDING_MODEL = 'text-embedding-ada-002'
    """Default model to do text embeddings."""

    EMBEDDING_URL = 'https://api.openai.com/v1/embeddings'
    """OpenAI embedding API URL"""

    URL = 'https://api.openai.com/v1/chat/completions'
    """OpenAI API API URL"""

    HEADERS = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai.api_key}",
               "api-key": f"{openai.api_key}",
               }
    """OpenAI API Headers"""

    MODEL_ROLE = "You are a research assistant that answers questions."
    """High level model role"""

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : None | str
            Optional specification of OpenAI model to use. Default is
            cls.DEFAULT_MODEL
        """
        self.model = model or self.DEFAULT_MODEL
        self.chat_messages = [{"role": "system", "content": self.MODEL_ROLE}]
        self.api_queue = None

    @staticmethod
    async def call_api(url, headers, request_json):
        """Make an asyncronous OpenAI API call.

        Parameters
        ----------
        url : str
            OpenAI API url, typically either:
                https://api.openai.com/v1/embeddings
                https://api.openai.com/v1/chat/completions
        headers : dict
            OpenAI API headers, typically:
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {openai.api_key}"}
        request_json : dict
            API data input, typically looks like this for chat completion:
                {"model": "gpt-3.5-turbo",
                 "messages": [{"role": "system", "content": "You do this..."},
                              {"role": "user", "content": "Do this: {}"}],
                 "temperature": 0.0}

        Returns
        -------
        out : dict
            API response in json format
        """

        out = None
        kwargs = dict(url=url, headers=headers, json=request_json)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(**kwargs) as response:
                    out = await response.json()

        except Exception as e:
            logger.debug(f'Error in OpenAI API call from '
                         f'`aiohttp.ClientSession().post(**kwargs)` with '
                         f'kwargs: {kwargs}')
            logger.exception('Error in OpenAI API call! Turn on debug logging '
                             'to see full query that caused error.')
            out = {'error': str(e)}

        return out

    async def call_api_async(self, url, headers, all_request_jsons,
                             rate_limit=40e3):
        """Use GPT to clean raw pdf text in parallel calls to the OpenAI API.

        NOTE: you need to call this using the await command in ipython or
        jupyter, e.g.: `out = await PDFtoTXT.clean_txt_async()`

        Parameters
        ----------
        url : str
            OpenAI API url, typically either:
                https://api.openai.com/v1/embeddings
                https://api.openai.com/v1/chat/completions
        headers : dict
            OpenAI API headers, typically:
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {openai.api_key}"}
        all_request_jsons : list
            List of API data input, one entry typically looks like this for
            chat completion:
                {"model": "gpt-3.5-turbo",
                 "messages": [{"role": "system", "content": "You do this..."},
                              {"role": "user", "content": "Do this: {}"}],
                 "temperature": 0.0}
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            gpt-3.5-turbo limit is 90k as of 4/2023, but we're using a large
            factor of safety (~1/2) because we can only count the tokens on the
            input side and assume the output is about the same count.

        Returns
        -------
        out : list
            List of API outputs where each list entry is a GPT answer from the
            corresponding message in the all_request_jsons input.
        """
        self.api_queue = ApiQueue(url, headers, all_request_jsons,
                                  rate_limit=rate_limit)
        out = await self.api_queue.run()
        return out

    def chat(self, query, temperature=0):
        """Have a continuous chat with the LLM including context from previous
        chat() calls stored as attributes in this class.

        Parameters
        ----------
        query : str
            Question to ask ChatGPT
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.

        Returns
        -------
        response : str
            Model response
        """

        self.chat_messages.append({"role": "user", "content": query})

        kwargs = dict(model=self.model,
                      messages=self.chat_messages,
                      temperature=temperature,
                      stream=False)
        if 'azure' in str(openai.api_type).lower():
            kwargs['engine'] = self.model

        response = openai.ChatCompletion.create(**kwargs)
        response = response["choices"][0]["message"]["content"]
        self.chat_messages.append({'role': 'assistant', 'content': response})

        return response

    def generic_query(self, query, model_role=None, temperature=0):
        """Ask a generic single query without conversation

        Parameters
        ----------
        query : str
            Question to ask ChatGPT
        model_role : str | None
            Role for the model to take, e.g.: "You are a research assistant".
            This defaults to self.MODEL_ROLE
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.

        Returns
        -------
        response : str
            Model response
        """

        model_role = model_role or self.MODEL_ROLE
        messages = [{"role": "system", "content": model_role},
                    {"role": "user", "content": query}]
        kwargs = dict(model=self.model,
                      messages=messages,
                      temperature=temperature,
                      stream=False)

        if 'azure' in str(openai.api_type).lower():
            kwargs['engine'] = self.model

        response = openai.ChatCompletion.create(**kwargs)
        response = response["choices"][0]["message"]["content"]
        return response

    async def generic_async_query(self, queries, model_role=None,
                                  temperature=0, rate_limit=40e3):
        """Run a number of generic single queries asynchronously
        (not conversational)

        NOTE: you need to call this using the await command in ipython or
        jupyter, e.g.: `out = await Summary.run_async()`

        Parameters
        ----------
        query : list
            Questions to ask ChatGPT (list of strings)
        model_role : str | None
            Role for the model to take, e.g.: "You are a research assistant".
            This defaults to self.MODEL_ROLE
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            gpt-3.5-turbo limit is 90k as of 4/2023, but we're using a large
            factor of safety (~1/2) because we can only count the tokens on the
            input side and assume the output is about the same count.

        Returns
        -------
        response : list
            Model responses with same length as query input.
        """

        model_role = model_role or self.MODEL_ROLE
        all_request_jsons = []
        for msg in queries:
            msg = [{'role': 'system', 'content': self.MODEL_ROLE},
                   {'role': 'user', 'content': msg}]
            req = {"model": self.model, "messages": msg,
                   "temperature": temperature}
            all_request_jsons.append(req)

        self.api_queue = ApiQueue(self.URL, self.HEADERS, all_request_jsons,
                                  rate_limit=rate_limit)
        out = await self.api_queue.run()

        for i, response in enumerate(out):
            choice = response.get('choices', [{'message': {'content': ''}}])[0]
            message = choice.get('message', {'content': ''})
            content = message.get('content', '')
            if not any(content):
                logger.error(f'Received no output for query {i + 1}!')
            else:
                out[i] = content

        return out

    @classmethod
    def get_embedding(cls, text):
        """Get the 1D array (list) embedding of a text string.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        embedding : list
            List of float that represents the numerical embedding of the text
        """
        kwargs = dict(url=cls.EMBEDDING_URL,
                      headers=cls.HEADERS,
                      json={'model': cls.EMBEDDING_MODEL,
                            'input': text})

        out = requests.post(**kwargs)
        embedding = out.json()

        try:
            embedding = embedding["data"][0]["embedding"]
        except Exception as exc:
            msg = ('Embedding request failed: {} {}'
                   .format(out.reason, embedding))
            logger.error(msg)
            raise RuntimeError(msg) from exc

        return embedding

    @staticmethod
    def count_tokens(text, model):
        """Return the number of tokens in a string.

        Parameters
        ----------
        text : str
            Text string to get number of tokens for
        model : str
            specification of OpenAI model to use (e.g., "gpt-3.5-turbo")

        Returns
        -------
        n : int
            Number of tokens in text
        """

        # Optional mappings for weird azure names to tiktoken/openai names
        tokenizer_aliases = {'gpt-35-turbo': 'gpt-3.5-turbo',
                             'gpt-4-32k': 'gpt-4-32k-0314'
                             }

        token_model = tokenizer_aliases.get(model, model)
        encoding = tiktoken.encoding_for_model(token_model)

        return len(encoding.encode(text))


class ApiQueue:
    """Class to manage the parallel API queue and submission"""

    def __init__(self, url, headers, request_jsons, rate_limit=40e3):
        """
        Parameters
        ----------
        url : str
            OpenAI API url, typically either:
                https://api.openai.com/v1/embeddings
                https://api.openai.com/v1/chat/completions
        headers : dict
            OpenAI API headers, typically:
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {openai.api_key}"}
        all_request_jsons : list
            List of API data input, one entry typically looks like this for
            chat completion:
                {"model": "gpt-3.5-turbo",
                 "messages": [{"role": "system", "content": "You do this..."},
                              {"role": "user", "content": "Do this: {}"}],
                 "temperature": 0.0}
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            gpt-3.5-turbo limit is 90k as of 4/2023, but we're using a large
            factor of safety (~1/2) because we can only count the tokens on the
            input side and assume the output is about the same count.
        """

        self.url = url
        self.headers = headers
        self.request_jsons = request_jsons
        self.rate_limit = rate_limit

        self.api_jobs = {}
        self.todo = [True] * len(self)
        self.out = [None] * len(self)

    def __len__(self):
        """Number of API calls to submit"""
        return len(self.request_jsons)

    def submit_jobs(self):
        """Submit a subset jobs asynchronously and hold jobs in the `api_jobs`
        attribute. Break when the `rate_limit` is exceeded."""
        token_count = 0
        for i, itodo in enumerate(self.todo):
            if (i not in self.api_jobs
                    and itodo
                    and token_count < self.rate_limit):
                request = self.request_jsons[i]
                model = request['model']
                token = ApiBase.count_tokens(str(request), model)
                token_count += token

                task = asyncio.create_task(ApiBase.call_api(self.url,
                                                            self.headers,
                                                            request))
                self.api_jobs[i] = task

                logger.debug('Submitted {} out of {}, '
                             'token count is at {} '
                             '(rate limit is {})'
                             .format(i + 1, len(self), token_count,
                                     self.rate_limit))

            elif token_count >= self.rate_limit:
                token_count = 0
                time.sleep(60)
                break

    async def collect_jobs(self):
        """Collect asyncronous API calls and API outputs. Store outputs in the
        `out` attribute."""
        complete = len(self) - sum(self.todo)
        for i, itodo in enumerate(self.todo):
            if itodo and i in self.api_jobs:
                task_out = await self.api_jobs[i]

                if 'error' in task_out:
                    logger.error('Received API error for task #{}: {}'
                                 .format(i + 1, task_out))
                else:
                    self.out[i] = task_out
                    self.todo[i] = False
                    complete = len(self) - sum(self.todo)

        logger.debug('Finished {} API calls, have {} left'
                     .format(complete, sum(self.todo)))

    async def run(self):
        """Run all asyncronous API calls.

        Returns
        -------
        out : list
            List of API call outputs with same ordering as `request_jsons`
            input.
        """

        logger.debug('Submitting async API calls...')

        while any(self.todo):
            self.submit_jobs()
            await self.collect_jobs()

        return self.out
