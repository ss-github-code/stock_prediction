function ticker_submit(event) {
    event.preventDefault();
    var data = new FormData($('#ticker_form').get(0));
    $.ajax({
        url: $(this).attr('action'),
        type: $(this).attr('method'),
        data: data,
        cache: false,
        processData: false,
        contentType: false,

        beforeSend: function(){
            $('.loading_gif_container').css('display', 'block');
        },

        complete: function(){
            $('.loading_gif_container').css('display', 'none');
        },

        success: function (data) {
            console.log('form submitted successfully')
            console.log(data)

            $('.table_container').css('display', 'block');
            $('#today_str').html(data['today_str'])
            $('#arma_cell').html(data['arma_prediction'][0])
            $('#arima_cell').html(data['arima_prediction'][0])
            $('#lstm_cell').html(data['lstm_prediction'][0])
            $('#lstm_w_sent_cell').html(data['lstm_w_sent_prediction'][0])

            $('#arma_mse_cell').html(data['arma_prediction'][1]['mse'])
            $('#arma_mae_cell').html(data['arma_prediction'][1]['mae'])
            $('#arma_mape_cell').html(data['arma_prediction'][1]['mape'])
            $('#arma_rmse_cell').html(data['arma_prediction'][1]['rmse'])

            $('#arima_mse_cell').html(data['arima_prediction'][1]['mse'])
            $('#arima_mae_cell').html(data['arima_prediction'][1]['mae'])
            $('#arima_mape_cell').html(data['arima_prediction'][1]['mape'])
            $('#arima_rmse_cell').html(data['arima_prediction'][1]['rmse'])

            $('#lstm_mse_cell').html(data['lstm_prediction'][1]['mse'])
            $('#lstm_mae_cell').html(data['lstm_prediction'][1]['mae'])
            $('#lstm_mape_cell').html(data['lstm_prediction'][1]['mape'])
            $('#lstm_rmse_cell').html(data['lstm_prediction'][1]['rmse'])

            $('#lstm_w_sent_mse_cell').html(data['lstm_w_sent_prediction'][1]['mse'])
            $('#lstm_w_sent_mae_cell').html(data['lstm_w_sent_prediction'][1]['mae'])
            $('#lstm_w_sent_mape_cell').html(data['lstm_w_sent_prediction'][1]['mape'])
            $('#lstm_w_sent_rmse_cell').html(data['lstm_w_sent_prediction'][1]['rmse'])            
        }
    });
    return false;
}

$(function () {
    $('#ticker_form').submit(ticker_submit);
});