#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """
    Input values for EXTRACTIVE text summarization
    """
    text: str = Field(..., example='John ate an apple', title='Text to summarize')
    # nb_sentences: int = Field(example= '3', title='Number of sentences to be selected in the final summary')

# class InferenceInputAbs(BaseModel):
#     """
#     Input values for ABSTRACTIVE text summarization
#     """
#     text: str = Field(..., example='John ate an apple', title='Text to summarize')
#     # min_length: int = Field(example= '75', title='Minimal number of words (tokens) to be generated')
#     # max_length: int = Field(example= '200', title='Maximal number of words (tokens) to be generated')
    
class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    summary: str = Field(..., example='245 Covid-19 cases have been recorded since last Tuesday.', title='Summarized text')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Whether there is error')
    results: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')