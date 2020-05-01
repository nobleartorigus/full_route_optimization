const { Router } = require('express')
const router = Router()
const { renderSignUpForm, renderLogInForm, login, signup, logout, renderAdminForm, createAdmin } = require('../controllers/users.controller')
const { isSuperAdmin } = require('../helpers/auth')
router.get('/users/signup', renderSignUpForm)

router.post('/users/signup', signup)

router.get('/admin/signup', isSuperAdmin, renderAdminForm)

router.post('/admin/signup', createAdmin)

router.get('/users/login', renderLogInForm)

router.post('/users/login', login)

router.get('/users/logout', logout)

module.exports = router