module scratchpad_controller_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int ADDR_WIDTH = $clog2(DEPTH);

  logic clk;
  logic rst_n;
  logic write_activations_en_i;
  logic [ADDR_WIDTH-1:0] activation_write_addr_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_i;
  logic write_weights_en_i;
  logic [ADDR_WIDTH-1:0] weight_write_addr_i;
  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_i;
  logic load_vector_en_i;
  logic [ADDR_WIDTH-1:0] activation_read_addr_i;
  logic [ADDR_WIDTH-1:0] weight_read_addr_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_west_o;
  logic signed [COLS*DATA_WIDTH-1:0] weights_north_o;
  logic vector_valid_o;

  scratchpad_controller #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dut (
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
    .activations_west_o(activations_west_o),
    .weights_north_o(weights_north_o),
    .vector_valid_o(vector_valid_o)
  );

  always #5 clk = ~clk;

  task automatic idle_controls;
    begin
      write_activations_en_i = 1'b0;
      write_weights_en_i = 1'b0;
      load_vector_en_i = 1'b0;
      activations_write_data_i = '0;
      weights_write_data_i = '0;
    end
  endtask

  task automatic write_activation_vector(
    input integer addr,
    input integer row0,
    input integer row1
  );
    begin
      idle_controls();
      activation_write_addr_i = addr[ADDR_WIDTH-1:0];
      activations_write_data_i[0 +: DATA_WIDTH] = row0;
      activations_write_data_i[DATA_WIDTH +: DATA_WIDTH] = row1;
      write_activations_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic write_weight_vector(
    input integer addr,
    input integer col0,
    input integer col1
  );
    begin
      idle_controls();
      weight_write_addr_i = addr[ADDR_WIDTH-1:0];
      weights_write_data_i[0 +: DATA_WIDTH] = col0;
      weights_write_data_i[DATA_WIDTH +: DATA_WIDTH] = col1;
      write_weights_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic load_and_expect(
    input integer act_addr,
    input integer weight_addr,
    input integer act0,
    input integer act1,
    input integer w0,
    input integer w1,
    input logic expected_valid
  );
    begin
      idle_controls();
      activation_read_addr_i = act_addr[ADDR_WIDTH-1:0];
      weight_read_addr_i = weight_addr[ADDR_WIDTH-1:0];
      load_vector_en_i = expected_valid;
      @(posedge clk);
      #1;
      if ($signed(activations_west_o[0 +: DATA_WIDTH]) !== act0 ||
          $signed(activations_west_o[DATA_WIDTH +: DATA_WIDTH]) !== act1 ||
          $signed(weights_north_o[0 +: DATA_WIDTH]) !== w0 ||
          $signed(weights_north_o[DATA_WIDTH +: DATA_WIDTH]) !== w1 ||
          vector_valid_o !== expected_valid) begin
        $fatal(1, "scratchpad_controller_tb failed");
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    activation_read_addr_i = '0;
    weight_read_addr_i = '0;
    idle_controls();
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    write_activation_vector(0, 1, 2);
    write_weight_vector(0, 5, 6);
    write_activation_vector(1, 3, 4);
    write_weight_vector(1, 7, 8);

    load_and_expect(0, 0, 1, 2, 5, 6, 1'b1);
    load_and_expect(1, 1, 3, 4, 7, 8, 1'b1);
    load_and_expect(1, 1, 3, 4, 7, 8, 1'b0);

    $display("scratchpad_controller_tb passed");
    $finish;
  end
endmodule
