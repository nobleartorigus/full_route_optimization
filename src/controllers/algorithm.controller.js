const path = require('path')
const Maps = require('../models/Maps')
const Associate = require('../models/Associates')

algorithmController = {}

algorithmController.createData = async (req, res) => {

    //console.log(req.user.city)
    const city = req.user.city

    const maps = await Maps.find({city}).lean()
    console.log(maps)
    res.render('routes_algorithm/routes_index', {maps} )
    //res.sendFile(path.resolve('src/views/routes_algorithm/maps/' + city))
}

algorithmController.renderData = async (req, res) => {
    //res.sendFile('routes_algorithm/maps/mapeo_asociados_ags.html')
    console.log(req.params.id)
    const id = req.params.id
    const map = await Maps.findById(id).lean()

    const url = map.url
    console.log(map.url)

    res.sendFile(path.resolve(url))
    //res.render('routes_algorithm/routes_maps', {url} )
}

algorithmController.mapForm = async (req, res) => {
    console.log(req.params)
    const id = req.params.id

    const mapurl = await Maps.findById(id).lean()
    console.log(mapurl)
    res.render('routes_algorithm/routes_maps', {mapurl} )
}

algorithmController.renderAssociates = async (req, res) => {
    console.log(req.user.city)
    const city = req.user.city
    //const Associates = await Associate.find({city}).lean()
    const Associates = await Associate.find({}).lean()
    res.render('associates/associates', { Associates })
}

algorithmController.deleteAssociate = async (req, res) => {
    await Associate.findByIdAndDelete(req.params.id)
    console.log('Deleting Associate')
    req.flash('success_message', 'Associate Deleted Succesfully')

    res.redirect('/associates')
}

module.exports = algorithmController