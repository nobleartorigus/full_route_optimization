const { Router } = require('express')
const router = Router()

const { createData, renderData, renderAssociates, deleteAssociate } = require('../controllers/algorithm.controller')
const { isAuthenticated } = require('../helpers/auth')

router.get('/routes', isAuthenticated, createData)

router.get('/maps/:id', isAuthenticated, renderData)

router.get('/associates', isAuthenticated, renderAssociates)

router.delete('/associates/delete/:id', isAuthenticated, deleteAssociate)

module.exports = router