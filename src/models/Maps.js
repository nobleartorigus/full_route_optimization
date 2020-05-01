const { Schema, model } = require('mongoose')

const mapsSchema = new Schema({
    country: {
        type: String,
        //required: true
    }, 
    city: {
        type: String,
        //required: true
    },
    url: {
        type: String,
        //required: true
    },
    user: {
        type: String,
        //required: true
    }
}, {
    timestamps: true
})

// mapsSchema.methods.saveToDB = async (user, country, city, url) => {
//     const newMap = new mapsSchema({user, country, city, url})
//     await newMap.save()
//     console.log(newMap)
//     return 0
// }

// mapsSchema.methods.saveToDB = async (user) => {
//     console.log(user)
//     return 0
// }


module.exports = model('Map', mapsSchema)