module tile_compute_unit #(
  parameter int DATA_WIDTH = 8,
  parameter int ACC_WIDTH = 32,
  parameter int ROWS = 2,
  parameter int COLS = 2,
  parameter int DEPTH = 4,
  parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
  input  logic clk,
  input  logic rst_n,
  input  logic write_activations_en_i,
  input  logic [ADDR_WIDTH-1:0] activation_write_addr_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_i,
  input  logic write_weights_en_i,
  input  logic [ADDR_WIDTH-1:0] weight_write_addr_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_i,
  input  logic load_vector_en_i,
  input  logic [ADDR_WIDTH-1:0] activation_read_addr_i,
  input  logic [ADDR_WIDTH-1:0] weight_read_addr_i,
  input  logic compute_en_i,
  input  logic clear_acc_i,
  output logic scratchpad_vector_valid_o,
  output logic signed [ROWS*COLS*ACC_WIDTH-1:0] psums_o,
  output logic [ROWS*COLS-1:0] valids_o
);
  logic signed [ROWS*DATA_WIDTH-1:0] activations_west;
  logic signed [COLS*DATA_WIDTH-1:0] weights_north;

  scratchpad_controller #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) scratchpad_inst (
    .clk(clk),
    .rst_n(rst_n),
    .write_activations_en_i(write_activations_en_i),
    .activation_write_addr_i(activation_write_addr_i),
    .activations_write_data_i(activations_write_data_i),
    .write_weights_en_i(write_weights_en_i),
    .weight_write_addr_i(weight_write_addr_i),
    .weights_write_data_i(weights_write_data_i),
    .load_vector_en_i(load_vector_en_i),
    .activation_read_addr_i(activation_read_addr_i),
    .weight_read_addr_i(weight_read_addr_i),
    .activations_west_o(activations_west),
    .weights_north_o(weights_north),
    .vector_valid_o(scratchpad_vector_valid_o)
  );

  systolic_tile #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS)
  ) tile_inst (
    .clk(clk),
    .rst_n(rst_n),
    .activations_west_i(activations_west),
    .weights_north_i(weights_north),
    .load_inputs_en(scratchpad_vector_valid_o),
    .compute_en(compute_en_i),
    .clear_acc(clear_acc_i),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );
endmodule
