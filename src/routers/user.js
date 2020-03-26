const express = require('express')
const mongoose = require('mongoose')
const User = require('../models/user')
const router = new express.Router()

// router.get('/users', (req, res) => {
//     res.send('Hello users')
// })

router.post('/users/', async (req, res) => {
    try {
        const user = await new User(req.body)
        user.save()
        res.status(200).send(user)
    } catch (e) {
        res.status(400).send(e)

    }
})

router.get('/users/', async (req, res) => {
    try {
        const users = await User.find({})
        res.status(200).send(users)
    } catch (e) {
        res.status(500).send()
    }
})



module.exports = router