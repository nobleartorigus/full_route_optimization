const express = require('express')
const mongoose = require('mongoose')
const Associate = require('../models/associates_data')
const router = new express.Router()

// router.get('/associates_data', (req, res) => {
//     res.send('Hello associates')
// })

router.get('/associates_data/', async (req, res) => {
    try {
        const associates_data = await Associate.find({})
        res.status(200).send(associates_data)
    } catch (e) {
        res.status(500).send()
    }
})

router.post('/associates_data', async (req, res) => {
    console.log('prueba')
    try {
        const prueba = await new Associate(req.body)
        prueba.save()
        res.status(200).send(prueba)
    } catch (e) {
        res.status(400).send(e)

    }
})


module.exports = router