const { Router } = require('express')
const router = Router()
const { renderSignUpForm, renderLogInForm, login, signup, logout } = require('../controllers/users.controller')

router.get('/users/signup', renderSignUpForm)

router.post('/users/signup', signup)

router.get('/users/login', renderLogInForm)

router.post('/users/login', login)

router.get('/users/logout', logout)

module.exports = router