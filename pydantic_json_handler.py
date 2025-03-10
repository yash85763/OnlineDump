import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Environment variables for API credentials
# USR_ID = os.getenv("USR_ID")
# USR_KEY = os.getenv("USR_KEY")

# Define your Pydantic models for structured output
class ProductFeature(BaseModel):
    name: str = Field(description="Name of the product feature")
    description: str = Field(description="Detailed description of the feature")
    importance: int = Field(description="Importance rating from 1-10", ge=1, le=10)

class ProductAnalysis(BaseModel):
    product_name: str = Field(description="Name of the product being analyzed")
    overall_rating: int = Field(description="Overall rating from 1-10", ge=1, le=10)
    summary: str = Field(description="Brief summary of the product analysis")
    features: List[ProductFeature] = Field(description="List of product features and their analyses")
    target_audience: List[str] = Field(description="List of target audience segments for this product")
    price_estimate: Optional[float] = Field(None, description="Estimated price point in USD if applicable")

# Method 1: Using JsonOutputParser with explicit formatting instructions
def get_json_with_output_parser(model_name="anthropic/claude-3-opus-20240229", usr_id=None, usr_key=None):
    # Initialize LLM
    if "anthropic" in model_name:
        llm = ChatAnthropic(
            model_name=model_name, 
            anthropic_api_key=usr_key,
            anthropic_user_id=usr_id,
            temperature=0
        )
    else:
        llm = ChatOpenAI(
            model_name=model_name, 
            openai_api_key=usr_key,
            openai_organization=usr_id,  # Using organization ID as the user ID
            temperature=0
        )
    
    # Create the parser
    parser = JsonOutputParser(pydantic_object=ProductAnalysis)
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a product analysis expert. Analyze the product and provide a structured response."),
        ("human", "Analyze the following product: {product}. {format_instructions}")
    ])
    
    # Add format instructions to the prompt
    prompt_with_parser = prompt.partial(format_instructions=parser.get_format_instructions())
    
    # Create the chain
    chain = prompt_with_parser | llm | parser
    
    # Run the chain
    result = chain.invoke({"product": "Electric standing desk with programmable height settings"})
    return result

# Method 2: Using Function Calling (more reliable for consistent JSON output)
def get_json_with_function_calling(model_name="anthropic/claude-3-opus-20240229", usr_id=None, usr_key=None):
    # Initialize LLM with function calling capability
    if "anthropic" in model_name:
        llm = ChatAnthropic(
            model_name=model_name, 
            anthropic_api_key=usr_key,
            anthropic_user_id=usr_id,
            temperature=0
        )
    else:
        llm = ChatOpenAI(
            model_name=model_name, 
            openai_api_key=usr_key,
            openai_organization=usr_id,  # Using organization ID as the user ID
            temperature=0
        )
    
    # Create function schema from Pydantic model
    functions = [
        {
            "name": "analyze_product",
            "description": "Analyze a product and return structured information",
            "parameters": ProductAnalysis.model_json_schema()
        }
    ]
    
    # Function to add function calling capability to the LLM
    if "anthropic" in model_name:
        llm_with_tools = llm.bind(tools=functions)
    else:
        llm_with_tools = llm.bind(functions=functions)
    
    # Create a parser to extract the function call
    parser = JsonOutputFunctionsParser()
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a product analysis expert. Analyze the product and provide a structured response."),
        ("human", "Analyze the following product: {product}")
    ])
    
    # Create the chain
    chain = prompt | llm_with_tools | parser
    
    # Run the chain
    result = chain.invoke({"product": "Electric standing desk with programmable height settings"})
    return result

# Method 3: Using Structured Output with function calling (LangChain 0.2.0+)
def get_json_with_structured_output(model_name="anthropic/claude-3-opus-20240229", usr_id=None, usr_key=None):
    # Initialize LLM
    if "anthropic" in model_name:
        llm = ChatAnthropic(
            model_name=model_name, 
            anthropic_api_key=usr_key,
            anthropic_user_id=usr_id,
            temperature=0
        )
    else:
        llm = ChatOpenAI(
            model_name=model_name, 
            openai_api_key=usr_key,
            openai_organization=usr_id,  # Using organization ID as the user ID
            temperature=0
        )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a product analysis expert. Analyze the product and provide a structured response."),
        ("human", "Analyze the following product: {product}")
    ])
    
    # Create the chain with structured output
    chain = prompt | llm.with_structured_output(ProductAnalysis)
    
    # Run the chain
    result = chain.invoke({"product": "Electric standing desk with programmable height settings"})
    return result

# Example usage:
if __name__ == "__main__":
    # Set your user ID and key
    usr_id = "your_user_id_here"  # Replace with your actual user ID
    usr_key = "your_user_key_here"  # Replace with your actual user key
    
    # Choose one of the methods
    # result = get_json_with_output_parser(usr_id=usr_id, usr_key=usr_key)
    # result = get_json_with_function_calling(usr_id=usr_id, usr_key=usr_key)
    result = get_json_with_structured_output(usr_id=usr_id, usr_key=usr_key)
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    # If you're using the structured output method, you can access fields directly
    if hasattr(result, "product_name"):
        print(f"Product name: {result.product_name}")
        print(f"Features:")
        for feature in result.features:
            print(f"- {feature.name}: {feature.importance}/10")