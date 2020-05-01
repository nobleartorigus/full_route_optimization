const usersController = {}
const User = require('../models/User')
const passport = require('passport')

usersController.renderSignUpForm = (req, res) => {
    res.render('users/signup')
}

usersController.signup = async (req, res) => {
    //res.send('signup')
    const errors = []
    console.log(req.body)
    const { name, email, password, city, confirm_password } = req.body
    if (password != confirm_password) {
        errors.push({text: 'Passwords do not match'})
    }

    if (password.length < 4) {
        errors.push({text: 'Passwords must be at least 4 characters'})
    }

    if (errors.length > 0) {
        res.render('users/signup', {
            errors,
            name,
            email
        })
    } else {
        const emailUser = await User.findOne({email: email})
        if (emailUser) {
            req.flash('error_message', 'The email provided is already in use')
            res.redirect('/users/signup')
        } 
        
        if (!city) {
            
            req.flash('error_message', 'Please choose a city')
            res.redirect('/users/signup')
        } else {
            const newUser = new User ({name, email, password, city, isAdmin: false})
            newUser.password = await newUser.encryptPassword(password)
            await newUser.save()
            req.flash('success_message', 'Register Succesfull')
            res.redirect('/users/login')
        }
    }
}

// Administradores

usersController.renderAdminForm = (req, res) => {
    res.render('users/admin_signup')
}

usersController.createAdmin = async (req, res) => {
    //const errors = []
    const email = req.body.email
    const isAdmin = req.body.isadmin
    console.log(email)
    console.log(isAdmin)

    const emailUser = await User.findOne({email: email})
    if (emailUser) {
        console.log(emailUser.isAdmin)
        console.log(emailUser.id)

        const userId = emailUser.id
        req.flash('success_message', 'The email provided is already in use')
        const updatedUser = await User.findByIdAndUpdate({_id: userId}, {isAdmin: isAdmin})
        res.redirect('/admin/signup')
        console.log(updatedUser)
    } else {
        req.flash('error_message', 'No user found')
        res.redirect('/admin/signup')
    }
    //const updateUser = await User.findByIdAndUpdate({email})

        
}

usersController.renderLogInForm = (req, res) => {
    res.render('users/login')
}

usersController.login = passport.authenticate('local', {
    failureRedirect: '/users/login',
    successRedirect: '/routes',
    failureFlash: true
})

// usersController.login = (req, res) => {
//     res.send('login')
// }

usersController.logout = (req, res) => {
    req.logout()
    req.flash('success_message', 'You are logged out now')
    res.redirect('/users/login')
}

module.exports = usersController