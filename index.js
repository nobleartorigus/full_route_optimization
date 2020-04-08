const express = require('express')
const upload = require('express-fileupload')

const userRouter = require('./src/routers/user')
const associatesRouter = require('./src/routers/associates')

require('./src/db/mongoose')

const app = express()

const http = require('http').Server(app)

app.use(express.json())
app.use(upload())
app.use(userRouter)
app.use(associatesRouter)

app.get('/', (req, res) => {
    res.send('Hello express')
})

//app.com
//app.com/help
//app.com/about


//PYTHON
    
    app.get('/python', (req, res) => { 
    
        const { spawn } = require('child_process'); 

        //Collect data from script
        const pyProg = spawn('python',['./src/hello.py']); 
    
        pyProg.stdout.on('data', function(data) { 
    
         console.log(data.toString()); 
         res.write(data); 

         //res.status(200).send(data)
         res.end(); 
    
        })
    }) 

    

//         // collect data from script

//         console.log('si hay proceso')
//         console.log(`Node Js recibio datos: ${data}`)
//         console.log(`Tipo de datos: ${typeof data}`)


//         // in close event we are sure that stream from child process is closed
//         // process.on('close', (code) => {
//         //     console.log(`child process close all stdio with code ${code}`);
//         //     res.status(200).send()
//         // })
//     })


//UPLOAD FILES

app.get("/uploads", (req, res) => {
    res.sendFile(__dirname+"/playground.html")
})

app.post('/uploads', (req, res) => {
    if(!req.files) {
        console.log("No file uploaded")
    }

    console.log(req.files)
    var file = req.files.filename, 
    filename = file.name
    file.mv("./src/uploads/"+filename, (err) => {
        if(err) {
            console.log(err)
            res.send("error ocurred")
        }

        res.send("Done! uploaded")
    })
})

//LISTEN SERVER


app.listen(3000, () => {
    console.log('Server is up on port 3000')
})

