const Note = require('../models/Note')
const NotesController = {}

NotesController.renderNoteForm = (req, res) => {
    //res.send('Note add')
    res.render('notes/new-note')
}

NotesController.createNewNote = async (req, res) => {
    const { title, description } = req.body
    newNote = new Note({title, description})
    await newNote.save()
    //res.send('New note created')
    req.flash('success_message', 'Note Added Succesfully')
    res.redirect('/notes')
}

NotesController.renderNotes = async (req, res) => {
    //res.send('Render All Notes')
    const notes = await Note.find().lean()
    res.render('notes/all-notes', { notes })
}

NotesController.renderEditForm = async (req, res) => {
    const note = await Note.findById(req.params.id).lean()
    console.log(note)
    res.render('notes/edit-note', { note })
    
}

NotesController.updateNote = async (req, res) => {
    console.log(req.body)
    const { title, description } =req.body
    //res.send('Update Note')
    await Note.findByIdAndUpdate(req.params.id, {title, description})
    req.flash('success_message', 'Note Updated Succesfully')
    res.redirect('/notes')
}

NotesController.deleteNote = async (req, res) => {
    //res.send('Deleting Note')
    await Note.findByIdAndDelete(req.params.id)
    console.log('Deleting Note')
    req.flash('success_message', 'Note Deleted Succesfully')

    res.redirect('/notes')
}

module.exports = NotesController