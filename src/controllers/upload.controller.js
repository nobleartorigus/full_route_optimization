const spawn = require('child_process').spawn

uploadController = {}

//UPLOAD FILES

uploadController.renderUploadForm = (req, res) => {
    res.render('uploads/upload')
}

uploadController.uploadFile = (req, res) => {
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

    const pythonProcess = spawn('python', ['route_algorithm/__init__.py', "src/uploads/"+filename])
    pythonProcess.stdout.on('data', (data) => {
        text = data.toString()

        // myjson = JSON.parse(text)

        // console.log(myjson)
        console.log(text)
        let success = 'success'

// --------------------------------------------------- BIG BUG ---------------------------------------------------

        if (text == success) {
            console.log('Failure')
            //req.flash('error_message', 'An error ocurred while creating your rote, please check if the document you uploaded is correct')
            //res.redirect('/upload')
        } else { 

            console.log('Success!')
            req.flash('success_message', 'File uploaded correctly, a new route has been created')
            res.redirect('/routes')
        }

        
    })


}

module.exports = uploadController