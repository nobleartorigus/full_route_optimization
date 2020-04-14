const { Schema, model }= require('mongoose')

const NoteSchema = new Schema({
    title: String,
    description: {
        type: String,
    }
}, {
    timestamps: true
})

module.exports = model('Note', NoteSchema)