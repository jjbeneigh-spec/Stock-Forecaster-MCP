# Stock Forecaster CSCI 357
# Author:
Jack Beneigh | Computer Science Senior '26 | Bucknell University
# Class:
CSCI 357 | Professor King

## Description
This notebook presents an end-to-end pipeline for short-horizon stock price forecasting
using a multi-layer Long Short-Term Memory (LSTM) neural network. The model ingests 60
trading days of OHLCV price data enriched with four technical indicators — RSI, MACD,
Bollinger Bands, and ATR, and predicts the next 5 closing prices for a given equity.

The trained model is deployed as a **Model Context Protocol (MCP) server, allowing
Claude Desktop to call it as a native tool and serve live forecasts in natural language.

My knowledge coming into this subject was small especially in regard to the MCP. It took a large amount of time to get Claude Desktop set up not only with my original test run with pre-determined price, but also once I had created my model which I integrated with Claude. 

### Goals and Motivation
I have been trying to expand my knowledge in stocks since soon enough I will have a job and will be making my own money. So when we were presented with this final project, I wanted to explore something that I will be able to learn from and also add to my portfolio. I do not have anything in "finance" on my portfolio so this was a way to expand my knowledge. I wanted to explore the MCP server because of the capabilities I have seen other people use it thus far. 

## Video
Check the file StockForecasterPresentation to see full explanation and motivation for the project. Also for live demonstration with Claude.

## Installation
The Direction for the Project will be in other file!

## Usage
THIS IS NOT A FINANCIAL TOOL THAT YOU SHOULD HAVE 100% FAITH IN!!!

This project is just practice and a very rough start at trying to understand MCP servers, Stock Forecasting and other techniques. Please do not use this and risk your money. Don't think I need to say it at all but be safe! :)))



You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

