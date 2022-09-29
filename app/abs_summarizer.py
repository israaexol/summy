from transformers import pipeline
from fastapi.logger import logger

async def abs_summarize(summarizer, text=None, file=None, min_length=75, max_length=300):
    if((file is None) and (text is not None)):
        to_tokenize = text
        summarized = summarizer(to_tokenize, min_length=min_length, max_length=max_length)
        print(summarized)
    elif((text is None) and (file is not None)):
        to_tokenize = await file.read()
        text = to_tokenize.decode("utf-8") 
        logger.info('Provided file')
        logger.info(f'content: {text}')
        summarized = summarizer(text, min_length=min_length, max_length=max_length)
    
    return summarized