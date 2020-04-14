const path = require('path')

algorithmController = {}

algorithmController.createData = (req, res) => {

    res.render('routes_algorithm/routes_index')
    //res.render('routes_algorithm/maps/mapeo_asociados_ags.hbs')
}

algorithmController.renderData = (req, res) => {
    //res.sendFile('routes_algorithm/maps/mapeo_asociados_ags.html')
    //console.log(req.body)
    res.sendFile(path.resolve('src/views/routes_algorithm/maps/mapeo_asociados_ags.html'))
}

module.exports = algorithmController