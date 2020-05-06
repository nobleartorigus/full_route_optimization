const { Router } = require('express')
const router = Router()

const { mapForm, createData, renderData, renderAssociates, deleteAssociate } = require('../controllers/algorithm.controller')
const { isAuthenticated } = require('../helpers/auth')

router.get('/routes', createData)

router.get('/turnos/:id' , mapForm)

router.get('/maps/:id', renderData)

router.get('/associates', isAuthenticated, renderAssociates)

router.delete('/associates/delete/:id', isAuthenticated, deleteAssociate)

module.exports = router