/* 
this follows setup for smithery's typescript SDK 

https://smithery.ai/server/@smithery-ai/server-sequential-thinking/api
*/ 

import express from 'express'
import {createTransport} from '@smithery/sdk/transport.js'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'

const app = express()
app.use(express.json())

let client: Client | null = null

// init endpoint -- allows setting up MCP client with any smithery URL
app.post('/init', async (req, res) => {
    try{
        const {smithery_url} = req.body
        if (!smithery_url) {
            return res.status(400).json({error: 'smithery URL required'})
        }
    const transport = createTransport(smithery_url)
    client = new Client({
        name: "MCP Bridge",
        version: '1.0.0'
    })
    await client.connect(transport)

    res.json({ status: 'connected', url: smithery_url })

    } catch (error) {
        res.status(500).json({ error: error.message })
    }
})

// middleware to check if client exists
const checkClient = (req, res, next) => {
    if (!client) {
        return res.status(400).json({
            error: "MCP client not initialized. Call /init first with smithery URL passed in body"
        })
    }
    next()
}

// list tools
app.get('/tools', checkClient, async(req, res) => {
    try {
        const tools = await client!.listTools()
        res.json(tools)
    } catch (error) {
        res.status(500).json({error: error.message})
    }
})

// call tool
app.post('/tools/:toolName', checkClient, async(req, res) => {
    try {
        const {toolName} = req.params
        const {params} = req.body
        const result = await client!.callTool(toolName, params)
        res.json(result)
    } catch (error) {
        res.status(500).json({ error: error.message})
    }
})

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`MCP Bridge runing on http://localhost:${PORT}`)
})