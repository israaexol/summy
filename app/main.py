import os
import sys
import traceback
import json
from transformers import pipeline

import uvicorn
from fastapi import FastAPI, Request, status, File, Form, UploadFile
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch

from abs_summarizer import abs_summarize
from ext_summarizer import generate_summary
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *
from config import CONFIG

# Initialize API Server
app = FastAPI(
    title="Summy",
    description="Text summarization using extractive and abstractive NLP techniques.",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the HuggingFace summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-base")
    summarizer.save_pretrained("/")

    # add model and other preprocess tools too app state
    app.package = {
        "summarizer": summarizer,
    }
    
#APIs for extractive summarization
@app.post('/summarize_ext',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def summarize_text_ext(request: Request, body: InferenceInput):
    """
    Perform an Extractive text summarization on data provided from text input
    """

    logger.info('API predict called')
    logger.info(f'input: {body}')

    # prepare input data
    text = body.text
    summarized = generate_summary(text, file=None)
    print(summarized)
    
    # if(body.nb_sentences is not None):
    #     nb_sentences = body.nb_sentences
    #     summarized = generate_summary()

    # prepare json for returning
    results = {
        'summary': summarized
    }

    logger.info(f'results: {results}')

    return {
        "error": False,
        "results": results
    }

@app.post('/summarize_ext_file',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def summarize_text_ext_file(request: Request, file: UploadFile = File(...)):
    """
    Perform an Extractive text summarization on data provided from file input (.txt)
    """

    logger.info('API predict called')
    logger.info(f'file input name: {file.filename}')

    # prepare input data

    summarized = await generate_summary(text=None, file=file)
    logger.info(f'summarized: {summarized}')

    
    # if(body.nb_sentences is not None):
    #     nb_sentences = body.nb_sentences
    #     summarized = generate_summary()

    # prepare json for returning
    results = {
        'summary': summarized
    }

    logger.info(f'results: {results}')

    return {
        "error": False,
        "results": results
    }
    
#####################################
#APIs for abstractive summarization
@app.post('/summarize_abs',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def summarize_text_abs(request: Request, body: InferenceInput):
    """
    Perform an Abstractive text summarization on data provided from text input
    """

    logger.info('API predict called')
    logger.info(f'input: {body}')

    # prepare input data
    text = body.text
    res = (await abs_summarize(app.package["summarizer"], text, file=None))[0]                             
    logger.info(f'res: {res}')
    summary = res['summary_text']
    
    # if(body.min_length is not None):
    #     min_length = body.min_length
    #     summarized = generate_summary()

    # prepare json for returning
    results = {
        'summary': summary
    }

    logger.info(f'results: {results}')

    return {
        "error": False,
        "results": results
    }
    
@app.post('/summarize_abs_file',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def summarize_text_abs_file(request: Request, file: UploadFile = File(...)):
    """
    Perform an Abstractive text summarization on data provided from file input (.txt)
    """

    logger.info('API predict called')
    logger.info(f'file input name: {file.filename}')

    # prepare input data

    res = await abs_summarize(app.package["summarizer"], text=None, file=file)                            
    logger.info(f'res: {res}')
    summary = res[0]['summary_text']
    
    # if(body.nb_sentences is not None):
    #     nb_sentences = body.nb_sentences
    #     summarized = generate_summary()

    # prepare json for returning
    results = {
        'summary': summary
    }

    logger.info(f'results: {results}')

    return {
        "error": False,
        "results": results
    }
    
##############################################

@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

@app.get("/")
def root():
    return {"data": "Welcome to Summy!"}

if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True, debug=True, log_config="log.ini"
                )