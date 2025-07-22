#!/usr/bin/env python3
“””
MCP Server for Financial Data Integration
Supports FactSet Data API and Yahoo Finance with fallbacks and error handling
“””

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import aiohttp
import yfinance as yf
from pydantic import BaseModel
import pandas as pd

# MCP imports

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
Resource,
Tool,
TextContent,
ImageContent,
EmbeddedResource,
CallToolResult,
ListResourcesResult,
ListToolsResult,
ReadResourceResult,
)

# Configure logging

logging.basicConfig(
level=logging.INFO,
format=’%(asctime)s - %(name)s - %(levelname)s - %(message)s’
)
logger = logging.getLogger(**name**)

# Configuration

class Config:
FACTSET_API_KEY = os.getenv(“FACTSET_API_KEY”)
FACTSET_BASE_URL = “https://api.factset.com”
YAHOO_FINANCE_ENABLED = True
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
CACHE_TTL = 300  # 5 minutes

class FactSetClient:
“”“FactSet API client with authentication and error handling”””

```
def __init__(self, api_key: str, base_url: str):
    self.api_key = api_key
    self.base_url = base_url
    self.session = None
    
async def __aenter__(self):
    self.session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT),
        headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    )
    return self
    
async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self.session:
        await self.session.close()

async def get_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
    """Get company fundamental data from FactSet"""
    endpoint = f"{self.base_url}/content/factset-fundamentals/v2/fundamentals"
    params = {
        "ids": symbol,
        "metrics": "SALES,NET_INCOME,TOTAL_DEBT,CASH_ST_INVEST",
        "periodicity": "ANN",
        "format": "JSON"
    }
    
    try:
        async with self.session.get(endpoint, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._process_fundamentals_response(data)
            elif response.status == 401:
                raise Exception("FactSet API authentication failed")
            elif response.status == 403:
                raise Exception("FactSet API access forbidden - check permissions")
            elif response.status == 429:
                raise Exception("FactSet API rate limit exceeded")
            else:
                error_text = await response.text()
                raise Exception(f"FactSet API error {response.status}: {error_text}")
                
    except aiohttp.ClientError as e:
        raise Exception(f"FactSet API connection error: {str(e)}")

async def get_price_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Get historical price data from FactSet"""
    endpoint = f"{self.base_url}/content/factset-prices/v1/prices"
    params = {
        "ids": symbol,
        "startDate": start_date,
        "endDate": end_date,
        "frequency": "D",
        "format": "JSON"
    }
    
    try:
        async with self.session.get(endpoint, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._process_price_response(data)
            else:
                error_text = await response.text()
                raise Exception(f"FactSet price API error {response.status}: {error_text}")
                
    except aiohttp.ClientError as e:
        raise Exception(f"FactSet price API connection error: {str(e)}")

def _process_fundamentals_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Process FactSet fundamentals API response"""
    if not data.get("data"):
        return {}
        
    processed = {}
    for item in data["data"]:
        metric = item.get("metric")
        value = item.get("value")
        date = item.get("date")
        
        if metric and value is not None:
            processed[metric] = {
                "value": value,
                "date": date,
                "currency": item.get("currency", "USD")
            }
            
    return processed

def _process_price_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Process FactSet price API response"""
    if not data.get("data"):
        return {}
        
    prices = []
    for item in data["data"]:
        prices.append({
            "date": item.get("date"),
            "open": item.get("open"),
            "high": item.get("high"), 
            "low": item.get("low"),
            "close": item.get("close"),
            "volume": item.get("volume")
        })
        
    return {"prices": prices}
```

class YahooFinanceClient:
“”“Yahoo Finance client with error handling”””

```
def __init__(self):
    pass

async def get_company_info(self, symbol: str) -> Dict[str, Any]:
    """Get company information from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract key information
        return {
            "symbol": symbol,
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "current_price": info.get("currentPrice"),
            "currency": info.get("currency", "USD")
        }
        
    except Exception as e:
        raise Exception(f"Yahoo Finance error for {symbol}: {str(e)}")

async def get_historical_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Get historical price data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise Exception(f"No historical data found for {symbol}")
        
        # Convert to list of dictionaries
        prices = []
        for date, row in hist.iterrows():
            prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"]) if pd.notna(row["Open"]) else None,
                "high": float(row["High"]) if pd.notna(row["High"]) else None,
                "low": float(row["Low"]) if pd.notna(row["Low"]) else None,
                "close": float(row["Close"]) if pd.notna(row["Close"]) else None,
                "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None
            })
        
        return {"prices": prices}
        
    except Exception as e:
        raise Exception(f"Yahoo Finance historical data error for {symbol}: {str(e)}")
```

class FinancialDataServer:
“”“Main MCP server class”””

```
def __init__(self):
    self.factset_available = bool(Config.FACTSET_API_KEY)
    self.yahoo_available = Config.YAHOO_FINANCE_ENABLED
    self.cache = {}  # Simple in-memory cache
    
    logger.info(f"FactSet available: {self.factset_available}")
    logger.info(f"Yahoo Finance available: {self.yahoo_available}")

async def get_company_data(self, symbol: str) -> Dict[str, Any]:
    """Get comprehensive company data with fallbacks"""
    cache_key = f"company_{symbol}"
    
    # Check cache
    if cache_key in self.cache:
        cache_entry = self.cache[cache_key]
        if datetime.now().timestamp() - cache_entry["timestamp"] < Config.CACHE_TTL:
            logger.info(f"Returning cached data for {symbol}")
            return cache_entry["data"]
    
    errors = []
    result = {"symbol": symbol, "data_sources": [], "errors": []}
    
    # Try FactSet first
    if self.factset_available:
        try:
            async with FactSetClient(Config.FACTSET_API_KEY, Config.FACTSET_BASE_URL) as client:
                fundamentals = await client.get_company_fundamentals(symbol)
                result["factset_fundamentals"] = fundamentals
                result["data_sources"].append("FactSet")
                logger.info(f"Successfully retrieved FactSet data for {symbol}")
        except Exception as e:
            error_msg = f"FactSet error: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)
    
    # Try Yahoo Finance as primary or fallback
    if self.yahoo_available:
        try:
            yahoo_client = YahooFinanceClient()
            company_info = await yahoo_client.get_company_info(symbol)
            result["yahoo_info"] = company_info
            result["data_sources"].append("Yahoo Finance")
            logger.info(f"Successfully retrieved Yahoo Finance data for {symbol}")
        except Exception as e:
            error_msg = f"Yahoo Finance error: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)
    
    # Check if we got any data
    if not result["data_sources"]:
        raise Exception(f"No data sources available for {symbol}. Errors: {'; '.join(errors)}")
    
    result["errors"] = errors
    
    # Cache the result
    self.cache[cache_key] = {
        "data": result,
        "timestamp": datetime.now().timestamp()
    }
    
    return result

async def get_price_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Get price data with fallbacks"""
    cache_key = f"prices_{symbol}_{period}"
    
    # Check cache
    if cache_key in self.cache:
        cache_entry = self.cache[cache_key]
        if datetime.now().timestamp() - cache_entry["timestamp"] < Config.CACHE_TTL:
            return cache_entry["data"]
    
    errors = []
    result = {"symbol": symbol, "period": period, "data_sources": [], "errors": []}
    
    # Calculate date range for FactSet
    end_date = datetime.now().strftime("%Y-%m-%d")
    if period == "1y":
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    elif period == "6mo":
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    elif period == "3mo":
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Try FactSet first
    if self.factset_available:
        try:
            async with FactSetClient(Config.FACTSET_API_KEY, Config.FACTSET_BASE_URL) as client:
                price_data = await client.get_price_data(symbol, start_date, end_date)
                result["factset_prices"] = price_data
                result["data_sources"].append("FactSet")
                logger.info(f"Successfully retrieved FactSet price data for {symbol}")
        except Exception as e:
            error_msg = f"FactSet price error: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)
    
    # Try Yahoo Finance
    if self.yahoo_available:
        try:
            yahoo_client = YahooFinanceClient()
            historical_data = await yahoo_client.get_historical_data(symbol, period)
            result["yahoo_prices"] = historical_data
            result["data_sources"].append("Yahoo Finance")
            logger.info(f"Successfully retrieved Yahoo Finance price data for {symbol}")
        except Exception as e:
            error_msg = f"Yahoo Finance price error: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)
    
    if not result["data_sources"]:
        raise Exception(f"No price data available for {symbol}. Errors: {'; '.join(errors)}")
    
    result["errors"] = errors
    
    # Cache the result
    self.cache[cache_key] = {
        "data": result,
        "timestamp": datetime.now().timestamp()
    }
    
    return result
```

# Initialize the MCP server

app = Server(“financial-data-server”)
financial_server = FinancialDataServer()

@app.list_resources()
async def list_resources() -> List[Resource]:
“”“List available resources”””
return [
Resource(
uri=“financial://company-data”,
name=“Company Financial Data”,
description=“Get comprehensive company financial data from multiple sources”,
mimeType=“application/json”
),
Resource(
uri=“financial://price-data”,
name=“Historical Price Data”,
description=“Get historical stock price data from multiple sources”,
mimeType=“application/json”
)
]

@app.read_resource()
async def read_resource(uri: str) -> str:
“”“Read a resource”””
if uri == “financial://company-data”:
return json.dumps({
“description”: “Company financial data endpoint”,
“usage”: “Use get_company_data tool with symbol parameter”,
“sources”: [“FactSet”, “Yahoo Finance”],
“fallbacks”: “Automatically falls back to available sources”
})
elif uri == “financial://price-data”:
return json.dumps({
“description”: “Historical price data endpoint”,
“usage”: “Use get_price_data tool with symbol and period parameters”,
“sources”: [“FactSet”, “Yahoo Finance”],
“periods”: [“1y”, “6mo”, “3mo”]
})
else:
raise ValueError(f”Unknown resource: {uri}”)

@app.list_tools()
async def list_tools() -> List[Tool]:
“”“List available tools”””
return [
Tool(
name=“get_company_data”,
description=“Get comprehensive company financial data including fundamentals, ratios, and company information”,
inputSchema={
“type”: “object”,
“properties”: {
“symbol”: {
“type”: “string”,
“description”: “Stock symbol (e.g., AAPL, MSFT, GOOGL)”
}
},
“required”: [“symbol”]
}
),
Tool(
name=“get_price_data”,
description=“Get historical stock price data with OHLCV information”,
inputSchema={
“type”: “object”,
“properties”: {
“symbol”: {
“type”: “string”,
“description”: “Stock symbol (e.g., AAPL, MSFT, GOOGL)”
},
“period”: {
“type”: “string”,
“enum”: [“1y”, “6mo”, “3mo”],
“description”: “Time period for historical data”,
“default”: “1y”
}
},
“required”: [“symbol”]
}
),
Tool(
name=“get_server_status”,
description=“Get the status of data sources and server configuration”,
inputSchema={
“type”: “object”,
“properties”: {}
}
)
]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
“”“Handle tool calls”””
try:
if name == “get_company_data”:
symbol = arguments.get(“symbol”, “”).upper()
if not symbol:
raise ValueError(“Symbol is required”)

```
        data = await financial_server.get_company_data(symbol)
        return [TextContent(
            type="text",
            text=json.dumps(data, indent=2, default=str)
        )]
    
    elif name == "get_price_data":
        symbol = arguments.get("symbol", "").upper()
        period = arguments.get("period", "1y")
        
        if not symbol:
            raise ValueError("Symbol is required")
        
        data = await financial_server.get_price_data(symbol, period)
        return [TextContent(
            type="text", 
            text=json.dumps(data, indent=2, default=str)
        )]
    
    elif name == "get_server_status":
        status = {
            "factset_available": financial_server.factset_available,
            "yahoo_available": financial_server.yahoo_available,
            "cache_entries": len(financial_server.cache),
            "timestamp": datetime.now().isoformat()
        }
        return [TextContent(
            type="text",
            text=json.dumps(status, indent=2)
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

except Exception as e:
    logger.error(f"Tool call error: {str(e)}")
    return [TextContent(
        type="text",
        text=json.dumps({
            "error": str(e),
            "tool": name,
            "arguments": arguments
        }, indent=2)
    )]
```

async def main():
“”“Main entry point”””
logger.info(“Starting Financial Data MCP Server”)

```
# Initialize the server with stdio transport
async with stdio_server() as streams:
    await app.run(
        streams[0], streams[1], 
        InitializationOptions(
            server_name="financial-data-server",
            server_version="1.0.0",
            capabilities=app.get_capabilities()
        )
    )
```

if **name** == “**main**”:
# Install required packages if needed
import subprocess
import sys

```
required_packages = [
    "mcp", 
    "aiohttp", 
    "yfinance", 
    "pandas", 
    "pydantic"
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

asyncio.run(main())
```