const helpers = {}
const User = require('../models/User')

helpers.isAuthenticated = (req, res, next) => {
    if (req.isAuthenticated()) {
        return next()
    }
    
    req.flash('error_message', 'Not Authorized')
    res.redirect('/users/login')
}


helpers.isAdmin = async (req, res, next) => {
    if (!req.user) {
        req.flash('error_message', 'Not Authorized')
        res.redirect('/users/login')
    }
    const session_id = req.user.id
    console.log(session_id)
    const user_session = await User.findById(session_id)
    if (user_session.isAdmin) {
        console.log('Accessed')
        return next()

    } else {
        req.flash('error_message', 'Access denied')
        res.redirect('/routes')
    }
}

helpers.isSuperAdmin = async (req, res, next) => {
    if (!req.user) {
        req.flash('error_message', 'Not Authorized')
        res.redirect('/users/login')
    }
    const session_id = req.user.id
    console.log(session_id)
    if (session_id === '5e9a09504f051a45d8e17dba') {
        console.log('Accessed')
        return next()

    } else {
        req.flash('error_message', 'Access denied')
        res.redirect('/routes')
    }
}

module.exports = helpers