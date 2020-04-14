const mongoose = require('mongoose')

//const MONGODB_URI = process.env.MONGODB_URI
// const ROUTE_OPTIMIZATION_MONGODB_HOST = process.env.ROUTE_OPTIMIZATION_MONGODB_HOST
// const ROUTE_OPTIMIZATION_MONGODB_DATABASE = process.env.ROUTE_OPTIMIZATION_MONGODB_DATABASE

const { ROUTE_OPTIMIZATION_MONGODB_HOST, ROUTE_OPTIMIZATION_MONGODB_DATABASE } = process.env
const MONGODB_URI = `mongodb://${ROUTE_OPTIMIZATION_MONGODB_HOST}/${ROUTE_OPTIMIZATION_MONGODB_DATABASE}`

mongoose.connect(MONGODB_URI, {
    useNewUrlParser: true,
    useCreateIndex: true,
    useCreateIndex: true
})
    .then(db => console.log('DB connected'))
    .catch(err => console.log(err))
