const mongoose = require('mongoose')

const Associates = mongoose.model('associates', {
//   'lat': 22.773475,
//   'lon': -102.572467,
//   'address': 'Aguascalientes, Mexico',
//   'count': 102,
    CP: {
        type: Number
    },
    lat: {
        type: Number
    },
    lon: {
        type: Number
    },
    address: {
        type: String
    },
    count: {
        type: Number
    }
})

// const associate = new Associates({
//     'CP': 65014,
//     'address': 'Aguascalientesn'
// })

// associate.save()

module.exports = Associates