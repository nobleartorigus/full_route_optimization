const { Schema, model } = require('mongoose')

const AssociateSchema = new Schema({
    cp: {
        type: Number,
        //required: true
    }, 
    lat: {
        type: Number,
        //required: true
    },
    lon: {
        type: Number,
        //required: true
    }
})


module.exports = model('Associate', AssociateSchema)