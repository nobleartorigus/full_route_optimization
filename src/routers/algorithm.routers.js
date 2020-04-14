const { Router } = require('express')
const router = Router()

const { createData, renderData } = require('../controllers/algorithm.controller')

router.get('/routes', createData)

router.get('/map', renderData)

module.exports = router