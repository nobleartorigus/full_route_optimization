const express = require('express')
const path = require('path')
const exphbs = require('express-handlebars')
const morgan = require('morgan')
const bodyParser = require('body-parser')
const override = require('method-override')
const flash = require('connect-flash')
const session = require('express-session')
const passport = require('passport')
const upload = require('express-fileupload')


// Initializers
const app = express()
require('./config/passport')

// Middlewares
app.use(bodyParser.urlencoded({extended: false}))
app.use(override('_method'))
//app.use(require('connect').bodyParser())
app.use(bodyParser.json())
app.use(morgan('dev'))
app.use(session({
    secret: 'secret',
    resave: true,
    saveUninitialized: true
}))
app.use(passport.initialize())
app.use(passport.session())
app.use(flash())
app.use(upload())

// Global Variables
app.use((req, res, next) => {
    res.locals.success_message = req.flash('success_message')
    res.locals.error_message = req.flash('error_message')
    res.locals.error = req.flash('error')
    res.locals.user = req.user || null

    next()
})

// Settings 
app.set('port', process.env.PORT || 4000) 
app.set('views',path.join(__dirname + '/views'))
app.engine('.hbs', exphbs({
    defaultLayout: 'main',
    layoutsDir: path.join(app.get('views'),'layouts'),
    partialsDir: path.join(app.get('views'), 'partials'),
    mapsDir: path.join(app.get('views'), 'routes_algorithm'),
    extname: '.hbs'
}))
app.set('view engine', '.hbs')

// Routes
// app.get('/', (req, res) => {
//     //res.send('Hello Express')
//     res.render('index')
// })

app.use(require('./routers/index.routers'))
app.use(require('./routers/notes.routers'))
app.use(require('./routers/users.routers'))
app.use(require('./routers/upload.routers'))
app.use(require('./routers/algorithm.routers'))


// Static Files
app.use(express.static(path.join(__dirname+'/public')))

module.exports = app