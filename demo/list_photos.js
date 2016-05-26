class PhotoList {
    constructor(photocropperObj) {
        this.limit = 12;
        this.photos = photocropperObj;
        this.$title = $('#title');
        this.$table = $('#table');
    }

    setTitle() {
        var theme = this.photos.theme;
        this.$title.html(`${theme} - tfPhotoClassifier`)
    }

    setTable() {
        var $table = $(document.createElement('table'));
        var label_expressions = this.photos.labels.answer_expression;
        var labels = this.photos.labels.labels;
        for (var i = 0; i < labels.length; i++) {
            var label_num = labels[i][0];
            var label = `label_${label_num}`;
            var photos = this.photos[label].items;
            
            var $tr = $(document.createElement('tr'));
            for (var j = 0; j < Math.min(this.limit, photos.length); j++) {
                var photo = photos[j];
                if (j === 0) {
                    var $td_ans = $(document.createElement('td'));
                    $td_ans.addClass('ans-label');
                    $td_ans.html(label_expressions[i]);
                    $tr.append($td_ans);
                }
                var $td = $(document.createElement('td'));
                var $img = $(document.createElement('div'));
                $img.css({
                    'background-image': `url(${photo.img_base64_cropped})`
                });
                $img.addClass('mini-photo');
                $td.append($img);
                $tr.append($td);
            }
            $table.append($tr);
        }
        this.$table.append($table);
    }
}

$(function () {
    var li = new PhotoList(photocropper);
    li.setTitle();
    li.setTable();
});