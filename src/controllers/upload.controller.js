const spawn = require('child_process').spawn
const Map = require('../models/Maps')
const User = require('../models/User')

uploadController = {}

//UPLOAD FILES

uploadController.renderUploadForm = async (req, res) => {

    //console.log(req.user.id)
    res.render('uploads/upload')
}

uploadController.uploadFile = async (req, res) => {

    global.prueba = 0

    // Destructuring data to save after process
    const { country, city } = req.body


    // Upload file
    if(!req.files) {
        req.flash('error_message', 'Please choose a file')
        res.redirect('/upload')
    }

    console.log(req.files.file)
    const file = req.files.file
    filename = file.name
    file.mv("./src/uploads/"+ filename, (err) => {
        if(err) {
            console.log(err)
            //res.send("error ocurred")
            req.flash('error_message', 'An error ocurred while uploading the file')
        }

        // res.send("Done! uploaded")
        req.flash('success_message', 'File uploaded correctly')
        //res.render('routes_algorithm/routes_index')
    })


    // Create the routes and maps
    const pythonProcess = spawn('python', ['route_algorithm/__init__.py', "src/uploads/"+filename, country, city])
    pythonProcess.stdout.on('data', async (data) => {
        text = data.toString()

        myjson = JSON.parse(text)

        console.log(myjson)
        console.log(myjson.status)
        //console.log(text)
        let success = 200

        if (myjson.status !== success) {
            console.log('Failure')

            req.flash('error_message', 'An error ocurred while creating your rote, please check if the document you uploaded is correct')
            res.redirect('/upload')
        } else { 

            console.log('Success!')
            req.flash('success_message', 'File uploaded correctly, a new route has been created')
            res.redirect('/routes')
            
        }

         //Save the data

        const user_id = req.user.id
        console.log('-------------------------------------------------')

        if (user_id === '5e99e5780bc6ca4dc0bfe250') {
            const newMap = new Map({country: country, city: city, url: myjson.maps.url_maps, user: 'superadmin'})
            console.log(newMap)
            await newMap.save()
        } else {
            const newMap = new Map({country: country, city: city, url: myjson.maps.url_maps, user: user_id})
            console.log(newMap)
            await newMap.save()
        }
       
    })

}


module.exports = uploadController